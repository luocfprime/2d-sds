import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from transformers import CLIPTextModel, CLIPTokenizer

from .. import instantiate, register
from ..utils.eval import get_clip_similarity
from ..utils.log import add_text_label, get_heatmap
from ..utils.misc import step_check
from ..utils.stable_diffusion import decode_latents, predict_noise
from ..utils.typings import DictConfig, List
from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


@register("sds")
class SDS(BaseAlgorithm):
    """
    Score Distillation Sampling
    https://arxiv.org/abs/2209.14988
    """

    @dataclass
    class Config(BaseAlgorithm.Config):
        name: str = "sds"
        device: str = "cuda"
        model_path: str = "stabilityai/stable-diffusion-2-1-base"
        prompt: str = ""
        neg_prompt: str = ""
        wt_schedule_cfg: DictConfig = field(default_factory=dict)
        guidance_scale: float = 100.0

        update_steps_per_iter: int = 1

        log_interval: int = 100
        log_sample_timesteps: List[int] = field(default_factory=lambda: [100, 500, 900])

        eval_clip_backbones: List[str] = field(
            default_factory=lambda: [
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14-336",
                "openai/clip-vit-large-patch14",
            ]
        )

    cfg: Config

    def __init__(self, cfg):
        self.cfg = self.validate_config(cfg)

        dtype = torch.float32  # torch.float32 by default
        device = torch.device(cfg.device)

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained(
            cfg.model_path, subfolder="vae", torch_dtype=dtype
        )
        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained(
            cfg.model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = CLIPTextModel.from_pretrained(
            cfg.model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        # 3. The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained(
            cfg.model_path, subfolder="unet", torch_dtype=dtype
        )
        # 4. Scheduler
        scheduler = DDPMScheduler.from_pretrained(
            cfg.model_path, subfolder="scheduler", torch_dtype=dtype
        )
        scheduler.set_timesteps(len(scheduler.betas))

        unet = unet.to(device)
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        # all variables in same device for scheduler.step()
        scheduler.betas = scheduler.betas.to(device)
        scheduler.alphas = scheduler.alphas.to(device)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.device = device

        # encode prompt
        text_input = tokenizer(
            [cfg.prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

        with torch.no_grad():
            self.text_embed = text_encoder(text_input.input_ids.to(device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [cfg.neg_prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            self.uncond_text_embed = text_encoder(uncond_input.input_ids.to(device))[0]

        self.wt_schedule = instantiate(cfg.wt_schedule_cfg)

        self.t_schedule = self.wt_schedule.t_schedule
        self.w_schedule = self.wt_schedule.w_schedule

    def compute_sds(self, latents, noise, t: torch.Tensor) -> dict:
        bsz = latents.shape[0]

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps=t)
        with torch.no_grad():
            noise_pred = predict_noise(
                unet=self.unet,
                noisy_latents=noisy_latents,
                text_embeddings=self.text_embed.expand(bsz, -1, -1),
                uncond_text_embeddings=self.uncond_text_embed.expand(bsz, -1, -1),
                t=t.repeat(bsz),
                scheduler=self.scheduler,
                guidance_scale=self.cfg.guidance_scale,
            )

        grad_raw = torch.nan_to_num(noise_pred - noise)

        return {
            "grad_raw": grad_raw,
            "noise_pred": noise_pred,
            "noise": noise,
            "noisy_latents": noisy_latents,
        }

    def step(self, step, rasterizer, writer):
        # 1. get latent
        latents = rasterizer.get_latents(self.vae)  # noqa
        bsz = latents.shape[0]
        # 2. add noise according to t schedule
        t = torch.tensor([self.t_schedule(step)]).to(self.device)
        # 3. compute sds
        noise = torch.randn_like(latents)

        loss_avg, grad_avg, grad_max, grad_min = 0, 0, 0, float("inf")
        loss_start, loss_end = None, None  # loss value of start and end of update
        for _ in range(self.cfg.update_steps_per_iter):
            # use same noise and update params multiple times
            (
                grad_raw,
                noise_pred,
                _,
                _,
            ) = self.compute_sds(latents, noise, t).values()

            grad = torch.nan_to_num(  # noqa
                grad_raw * self.w_schedule(step)  # apply weight schedule
            )

            target = (latents - grad).detach()
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / bsz

            # 4. update
            loss.backward()
            rasterizer.step(step)

            loss_avg += loss.item()
            grad_avg += grad.abs().mean().item()
            grad_max = max(grad_max, grad.abs().max().item())
            grad_min = min(grad_min, grad.abs().min().item())

            if loss_start is None:
                loss_start = loss.item()

            loss_end = loss.item()

        loss_avg /= self.cfg.update_steps_per_iter  # noqa
        grad_avg /= self.cfg.update_steps_per_iter

        # 5. log
        writer.add_scalar("loss/main", loss_avg, step)
        writer.add_scalar("loss/unscaled", loss_avg / self.w_schedule(step) ** 2, step)
        writer.add_scalar("loss/start", loss_start, step)
        writer.add_scalar("loss/end", loss_end, step)

        writer.add_scalar("grad/max", grad_max, step)
        writer.add_scalar("grad/mean", grad_avg, step)
        writer.add_scalar("grad/min", grad_min, step)

        writer.add_scalar("schedule/t_schedule", self.t_schedule(step), step)
        writer.add_scalar("schedule/w_schedule", self.w_schedule(step), step)

        if step_check(step, self.cfg.log_interval, run_at_zero=True):
            self.eval_step(step, rasterizer, writer)

    @torch.no_grad()
    def eval_step(self, step, rasterizer, writer):
        latents = rasterizer.get_latents(self.vae)
        bsz = latents.shape[0]

        rendered_images = decode_latents(
            latents, self.vae
        )  # 1. original rendered images
        writer.add_image(
            f"visualization/rendered", make_grid(rendered_images, nrow=bsz), step
        )

        images_x0s = []  # x0 at different sample timesteps
        heatmaps = []
        for t in self.cfg.log_sample_timesteps:  # sample with to different timesteps
            noise = torch.randn_like(latents)

            (grad_raw, noise_pred, noise, noisy_latents) = self.compute_sds(
                latents, noise, torch.tensor([t]).to(self.device)
            ).values()

            pred_latents_x0 = self.scheduler.step(
                noise_pred, t, noisy_latents
            ).pred_original_sample  # noqa
            images_x0 = decode_latents(pred_latents_x0, self.vae)

            grad = torch.nan_to_num(
                grad_raw * self.w_schedule(step)  # apply weight schedule
            )
            grad_abs = F.interpolate(
                grad.abs().mean(1, keepdim=True), size=rendered_images.shape[-2:]
            ).cpu()
            heatmap = get_heatmap(grad_abs, size=rendered_images.shape[-2:])

            images_x0s.append(images_x0)
            heatmaps.append(heatmap)

        def make_labeled_grid(images, labels):  # noqa
            images = torch.cat(images, dim=0)
            return add_text_label(make_grid(images, nrow=bsz), labels)

        writer.add_image(
            f"visualization/predict_x0_pseudo_gt",
            make_labeled_grid(
                images_x0s, [f"t={t}" for t in self.cfg.log_sample_timesteps]
            ),
            step,
        )
        writer.add_image(
            f"visualization/grad",
            make_labeled_grid(
                heatmaps, [f"t={t}" for t in self.cfg.log_sample_timesteps]
            ),
            step,
        )

        # eval clip score
        for clip_name in self.cfg.eval_clip_backbones:
            clip_similarity = get_clip_similarity(clip_name).to_device(self.device)
            clip_score = 0
            for img in rendered_images:
                rgb_pil = ToPILImage()(img)
                clip_score += clip_similarity.compute_text_img_similarity(
                    rgb_pil, self.cfg.prompt
                )
            clip_score /= len(rendered_images)
            writer.add_scalar(f"clip_score/{clip_name}", clip_score, step)
