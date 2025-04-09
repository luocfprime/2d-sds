"""
Some implementations are referenced from the official repo: https://github.com/KAIST-Visual-AI-Group/PDS
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa
import torchvision.transforms as T  # noqa
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import load_image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from transformers import CLIPTextModel, CLIPTokenizer

from .. import instantiate, register
from ..utils.eval import get_clip_similarity
from ..utils.log import add_text_label, get_heatmap
from ..utils.misc import step_check
from ..utils.stable_diffusion import decode_latents, encode_images, predict_noise
from ..utils.typings import DictConfig, List
from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


@register("pds")
class PDS(BaseAlgorithm):
    """
    Posterior Distillation Sampling
    https://openaccess.thecvf.com/content/CVPR2024/papers/Koo_Posterior_Distillation_Sampling_CVPR_2024_paper.pdf
    """

    @dataclass
    class Config(BaseAlgorithm.Config):
        name: str = "pds"
        device: str = "cuda"
        model_path: str = "stabilityai/stable-diffusion-2-1-base"

        height: int = 512
        width: int = 512

        source_prompts: List[str] = field(default_factory=lambda: [""])
        target_prompts: List[str] = field(default_factory=lambda: [""])
        source_neg_prompts: List[str] = field(default_factory=lambda: [""])
        target_neg_prompts: List[str] = field(default_factory=lambda: [""])

        source_images_paths: List[str] = field(default_factory=lambda: [""])
        wt_schedule_cfg: DictConfig = field(default_factory=lambda: dict())
        guidance_scale: float = 100.0

        update_steps_per_iter: int = 1

        log_interval: int = 200
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

        self.source_images = [load_image(p) for p in cfg.source_images_paths]

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

        transforms = T.Compose([T.Resize((cfg.height, cfg.width)), T.ToTensor()])

        image_tensors = torch.stack([transforms(im) for im in self.source_images])

        self.source_latents = encode_images(image_tensors, vae)

        # encode prompt
        assert len(cfg.source_prompts) == len(
            cfg.target_prompts
        ), "Number of source and target prompts must be the same"

        source_text_input = tokenizer(
            list(cfg.source_prompts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        target_text_input = tokenizer(
            list(cfg.target_prompts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

        with torch.no_grad():
            self.source_text_embed = text_encoder(
                source_text_input.input_ids.to(device)
            )[0]
            self.target_text_embed = text_encoder(
                target_text_input.input_ids.to(device)
            )[0]

        source_max_length = source_text_input.input_ids.shape[-1]
        if len(cfg.source_neg_prompts) == 1:  # if has only one neg prompt, expand
            source_neg_prompt = [cfg.source_neg_prompts[0]] * len(cfg.source_prompts)
        else:
            source_neg_prompt = cfg.source_neg_prompts
        source_uncond_input = tokenizer(
            source_neg_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=source_max_length,
        )

        target_max_length = target_text_input.input_ids.shape[-1]
        if len(cfg.target_neg_prompts) == 1:  # if has only one neg prompt, expand
            target_neg_prompt = [cfg.target_neg_prompts[0]] * len(cfg.target_prompts)
        else:
            target_neg_prompt = list(cfg.target_neg_prompts)

        target_uncond_input = tokenizer(
            target_neg_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=target_max_length,
        )

        with torch.no_grad():
            self.source_uncond_text_embed = text_encoder(
                source_uncond_input.input_ids.to(device)
            )[0]
            self.target_uncond_text_embed = text_encoder(
                target_uncond_input.input_ids.to(device)
            )[0]

        self.wt_schedule = instantiate(cfg.wt_schedule_cfg)

        self.t_schedule = self.wt_schedule.t_schedule
        self.w_schedule = self.wt_schedule.w_schedule

    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        Implementation referenced from https://github.com/KAIST-Visual-AI-Group/PDS
        """
        device = self.device
        beta_t = self.scheduler.betas[t].to(device)
        alpha_t = self.scheduler.alphas[t].to(device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(
            alpha_bar_t
        )
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        mean_func = c0 * pred_x0 + c1 * xt
        return mean_func

    @torch.no_grad()
    def compute_pds(
        self, latents, noise, noise_t_prev, t: torch.Tensor, t_prev: torch.Tensor
    ) -> dict:
        """
        Compute gradient using PDS
        Implementation referenced from https://github.com/KAIST-Visual-AI-Group/PDS
        Args:
            latents: current rendered latent
            noise:
            noise_t_prev:
            t:
            t_prev:

        Returns: dict containing grad tensor

        """
        bsz = latents.shape[0]

        beta_t = self.scheduler.betas[t].to(self.device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(self.device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(self.device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        zts = dict()
        mus = dict()
        for latent, cond_text_embedding, uncond_embedding, name in zip(
            [latents, self.source_latents],
            [self.target_text_embed, self.source_text_embed],
            [self.target_uncond_text_embed, self.source_uncond_text_embed],
            ["tgt", "src"],
        ):
            latents_noisy = self.scheduler.add_noise(latent, noise, t)

            noise_pred = predict_noise(
                unet=self.unet,
                noisy_latents=latents_noisy,
                text_embeddings=cond_text_embedding.expand(bsz, -1, -1),
                uncond_text_embeddings=uncond_embedding.expand(bsz, -1, -1),
                t=t.repeat(bsz),
                scheduler=self.scheduler,
                guidance_scale=self.cfg.guidance_scale,
            )

            x_t_prev = self.scheduler.add_noise(latent, noise_t_prev, t_prev)
            mu = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
            zt = (x_t_prev - mu) / sigma_t

            mus[name] = mu
            zts[name] = zt

        grad_raw = zts["tgt"] - zts["src"]
        grad_raw = torch.nan_to_num(grad_raw)

        return {"grad_raw": grad_raw, "source_mu": mus["src"], "target_mu": mus["tgt"]}

    def get_t_prev(self, t: torch.Tensor):
        """
        Get previous timestep from self.scheduler
        Args:
            t: shape [] or [1]

        Returns:
            t_prev: torch.Tensor, shape [], device same as t
        """
        timesteps = self.scheduler.timesteps.to(t.device)
        idx = timesteps.eq(t).nonzero(as_tuple=True)[0][0]
        t_prev = (
            timesteps[idx - 1] if idx > 1 else timesteps[1]
        )  # prevent alpha_bar_t_prev out of bound
        return t_prev

    def step(self, step, rasterizer, writer):
        # 1. get latent
        latents = rasterizer.get_latents(self.vae)  # noqa
        bsz = latents.shape[0]

        # 2. add noise according to t schedule
        t = torch.tensor([self.t_schedule(step)]).to(self.device)
        t_prev = self.get_t_prev(t).to(self.device)
        # 3. compute sds
        noise = torch.randn_like(latents)
        noise_t_prev = torch.randn_like(latents)

        loss_avg, grad_avg, grad_max, grad_min = 0, 0, 0, float("inf")
        for _ in range(self.cfg.update_steps_per_iter):
            # use same noise and update params multiple times (experimental)
            (grad_raw, *_) = self.compute_pds(
                latents, noise, noise_t_prev, t, t_prev
            ).values()

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

        loss_avg /= self.cfg.update_steps_per_iter  # noqa
        grad_avg /= self.cfg.update_steps_per_iter

        # 5. log
        writer.add_scalar("loss/main", loss_avg, step)
        writer.add_scalar("loss/unscaled", loss_avg / self.w_schedule(step) ** 2, step)

        writer.add_scalar("grad/max", grad_max, step)
        writer.add_scalar("grad/mean", grad_avg, step)
        writer.add_scalar("grad/min", grad_min, step)

        writer.add_scalar("schedule/t_schedule", self.t_schedule(step), step)
        writer.add_scalar("schedule/w_schedule", self.w_schedule(step), step)

        if step_check(step, self.cfg.log_interval, run_at_zero=True):
            self.eval_step(step, rasterizer, writer)

    @torch.no_grad()
    def eval_step(self, step, rasterizer, writer):
        latents = rasterizer.get_latents(self.vae)  # noqa
        bsz = latents.shape[0]

        rendered_images = decode_latents(
            latents, self.vae
        )  # 1. original rendered images
        writer.add_image(
            f"visualization/rendered", make_grid(rendered_images, nrow=bsz), step
        )

        source_images = decode_latents(self.source_latents, self.vae)

        heatmaps = []
        mus = {"src": [], "tgt": []}
        for t in self.cfg.log_sample_timesteps:  # sample with to different timesteps
            noise = torch.randn_like(latents)
            noise_t_prev = torch.randn_like(latents)

            (
                grad_raw,
                source_mu,
                target_mu,
            ) = self.compute_pds(
                latents,
                noise,
                noise_t_prev,
                torch.tensor([t]).to(self.device),
                self.get_t_prev(torch.tensor([t])).to(self.device),
            ).values()

            for mu, name in zip([source_mu, target_mu], ["src", "tgt"]):
                decoded_mu = decode_latents(mu, self.vae)
                mus[name].append(decoded_mu)

            grad = torch.nan_to_num(
                grad_raw * self.w_schedule(step)  # apply weight schedule
            )
            grad_abs = F.interpolate(
                grad.abs().mean(1, keepdim=True), size=rendered_images.shape[-2:]
            ).cpu()
            heatmap = get_heatmap(grad_abs, size=rendered_images.shape[-2:])

            heatmaps.append(heatmap)

        def make_labeled_grid(images, labels):  # noqa
            images = torch.cat(images, dim=0)
            return add_text_label(make_grid(images, nrow=bsz), labels)

        writer.add_image(
            f"visualization/src_mu",
            make_labeled_grid(
                mus["src"], [f"t={t}" for t in self.cfg.log_sample_timesteps]
            ),
            step,
        )
        writer.add_image(
            f"visualization/tgt_mu",
            make_labeled_grid(
                mus["tgt"], [f"t={t}" for t in self.cfg.log_sample_timesteps]
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
            clip_score = 0.0
            clip_directional_score = 0.0
            for i, (src_img, tgt_img) in enumerate(zip(source_images, rendered_images)):
                src_pil = ToPILImage()(src_img)
                tgt_pil = ToPILImage()(tgt_img)
                result = clip_similarity.compute_similarity(
                    src_img=src_pil,
                    tgt_img=tgt_pil,
                    src_prompt=self.cfg.source_prompts[i],
                    tgt_prompt=self.cfg.target_prompts[i],
                )
                clip_score += result["sim_1"]
                clip_directional_score += result["sim_direction"]

            clip_score /= len(rendered_images)
            clip_directional_score /= len(rendered_images)

            writer.add_scalar(f"clip_score/{clip_name}", clip_score, step)
            writer.add_scalar(
                f"clip_directional_score/{clip_name}", clip_directional_score, step
            )
