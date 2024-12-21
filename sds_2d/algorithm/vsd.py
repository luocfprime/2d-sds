import gc
import logging
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F  # noqa
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from transformers import CLIPTextModel, CLIPTokenizer

from .base import BaseAlgorithm
from .. import register, instantiate
from ..utils.eval import get_clip_similarity
from ..utils.log import get_heatmap, add_text_label
from ..utils.lora import set_lora_
from ..utils.misc import step_check
from ..utils.optimizer import get_optimizer
from ..utils.stable_diffusion import predict_noise, predict_noise_no_cfg, decode_latents
from ..utils.typings import DictConfig, List

logger = logging.getLogger(__name__)


@register("vsd")
class VSD(BaseAlgorithm):
    """
    Variational Score Distillation
    https://arxiv.org/abs/2305.16213
    """

    @dataclass
    class Config(BaseAlgorithm.Config):
        name: str = "vsd"
        device: str = "cuda"
        model_path: str = "stabilityai/stable-diffusion-2-1-base"
        prompt: str = ""
        neg_prompt: str = ""
        wt_schedule_cfg: DictConfig = field(default_factory=dict)
        guidance_scale: float = 7.5

        lora_scale: float = 1.0
        phi_update_step: int = 1
        phi_batch_size: int = 1
        optimizer: DictConfig = field(default_factory=dict)  # {name: str, opt_args: dict}

        log_interval: int = 200
        log_sample_timesteps: List[int] = field(default_factory=lambda: [100, 500, 900])

        eval_clip_backbones: List[str] = field(default_factory=lambda: [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14"
        ])

    cfg: Config

    def __init__(self, cfg):
        self.cfg = self.validate_config(cfg)

        dtype = torch.float32  # torch.float32 by default
        device = torch.device(cfg.device)

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained(cfg.model_path, subfolder="vae", torch_dtype=dtype)
        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer", torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder", torch_dtype=dtype)
        # 3. The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained(cfg.model_path, subfolder="unet", torch_dtype=dtype)
        # 4. Scheduler
        scheduler = DDPMScheduler.from_pretrained(cfg.model_path, subfolder="scheduler", torch_dtype=dtype)
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
        self.scheduler = scheduler
        self.device = device

        # 5. prepare phi model
        unet_lora, unet_lora_layers = set_lora_(unet)  # note that unet has been modified in-place
        phi_params = list(unet_lora_layers.parameters())

        self._unet_lora = unet_lora

        # 6. optimizer for phi model
        self.phi_optimizer = get_optimizer(cfg.optimizer, parameters=phi_params)

        # encode prompt
        text_input = tokenizer([cfg.prompt],
                               return_tensors="pt", padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length)

        with torch.no_grad():
            self.text_embed = text_encoder(text_input.input_ids.to(device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([cfg.neg_prompt],
                                 return_tensors="pt", padding="max_length", truncation=True,
                                 max_length=max_length)

        with torch.no_grad():
            self.uncond_text_embed = text_encoder(uncond_input.input_ids.to(device))[0]

        self.wt_schedule = instantiate(cfg.wt_schedule_cfg)

        self.t_schedule = self.wt_schedule.t_schedule
        self.w_schedule = self.wt_schedule.w_schedule

    @property
    def unet(self):
        return partial(self._unet_lora, cross_attention_kwargs={"scale": 0})  # disable lora

    @property
    def unet_phi(self):
        return partial(self._unet_lora, cross_attention_kwargs={"scale": self.cfg.lora_scale})  # enable lora

    def compute_vsd(self, latents, t: torch.Tensor) -> dict:
        bsz = latents.shape[0]

        # add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps=t)

        with torch.no_grad():
            # forward pretrained unet
            noise_pred = predict_noise(unet=self.unet,
                                       noisy_latents=noisy_latents,
                                       text_embeddings=self.text_embed.expand(bsz, -1, -1),
                                       uncond_text_embeddings=self.uncond_text_embed.expand(bsz, -1, -1),
                                       t=t.repeat(bsz),
                                       scheduler=self.scheduler,
                                       guidance_scale=self.cfg.guidance_scale
                                       )

            # forward phi model w/o cfg
            noise_pred_phi = predict_noise_no_cfg(unet=self.unet_phi,
                                                  noisy_latents=noisy_latents,
                                                  text_embeddings=self.text_embed.expand(bsz, -1, -1),
                                                  t=t.repeat(bsz),
                                                  scheduler=self.scheduler
                                                  )

        grad_raw = torch.nan_to_num(noise_pred - noise_pred_phi)

        return {
            "grad_raw": grad_raw,
            "noise_pred": noise_pred,
            "noise_pred_phi": noise_pred_phi,
            "noise": noise,
            "noisy_latents": noisy_latents
        }

    def step(self, step, rasterizer, writer):
        # 1. get latent
        latents = rasterizer.get_latents(self.vae)
        bsz = latents.shape[0]
        # 2. add noise according to t schedule
        t = torch.tensor([self.t_schedule(step)]).to(self.device)

        # 3. compute vsd
        (
            grad_raw,
            noise_pred,
            noise_pred_phi,
            _,
            _
        ) = self.compute_vsd(latents, t).values()

        grad = torch.nan_to_num(  # noqa
            grad_raw * self.w_schedule(step)  # apply weight schedule
        )

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / bsz

        # 4. update
        loss.backward()
        rasterizer.step(step)

        gc.collect()
        torch.cuda.empty_cache()

        # 5. update phi
        loss_phi = torch.tensor(0.0)
        for _ in range(self.cfg.phi_update_step):
            self.phi_optimizer.zero_grad()

            latents = rasterizer.get_latents(self.vae).detach()

            assert latents.shape[0] >= self.cfg.phi_batch_size, \
                f"batch size to train phi model should be less than or equal to rasterizer batch size {latents.shape[0]}"  # noqa

            latents = latents[torch.randperm(latents.shape[0])[:self.cfg.phi_batch_size]]  # random sample batch
            phi_bsz = latents.shape[0]

            t_phi = self.scheduler.timesteps[torch.randint(1, len(self.scheduler.timesteps),  # t=1000 cause index error
                                                           size=(phi_bsz,))].to(self.device)  # (phi_bsz,)

            noise_phi = torch.randn_like(latents)
            noisy_latents_phi = self.scheduler.add_noise(latents, noise_phi, timesteps=t_phi)

            noise_pred_phi = predict_noise_no_cfg(unet=self.unet_phi,
                                                  noisy_latents=noisy_latents_phi,
                                                  text_embeddings=self.text_embed.expand(phi_bsz, -1, -1),
                                                  t=t_phi,
                                                  scheduler=self.scheduler,
                                                  )

            loss_phi = 0.5 * F.mse_loss(noise_pred_phi, noise_phi, reduction="mean")
            loss_phi.backward()
            self.phi_optimizer.step()

        loss_phi /= self.cfg.phi_update_step

        # 6. log
        writer.add_scalar("loss/main", loss.item(), step)
        writer.add_scalar("loss/unscaled", loss.item() / self.w_schedule(step) ** 2, step)
        writer.add_scalar("loss/phi", loss_phi.item(), step)

        writer.add_scalar("grad/max", grad.abs().max().item(), step)
        writer.add_scalar("grad/mean", grad.abs().mean().item(), step)
        writer.add_scalar("grad/min", grad.abs().min().item(), step)

        writer.add_scalar("schedule/t_schedule", self.t_schedule(step), step)
        writer.add_scalar("schedule/w_schedule", self.w_schedule(step), step)

        if step_check(step, self.cfg.log_interval, run_at_zero=True):
            self.eval_step(step, rasterizer, writer)

    @torch.no_grad()
    def eval_step(self, step, rasterizer, writer):
        latents = rasterizer.get_latents(self.vae)
        bsz = latents.shape[0]

        rendered_images = decode_latents(latents, self.vae)  # 1. original rendered images
        writer.add_image(f"visualization/rendered", make_grid(rendered_images, nrow=bsz), step)

        images_x0s = []  # x0 at different sample timesteps
        images_phi_x0s = []  # phi unet predicted x0 images
        images_x0_pseudo_gts = []
        heatmaps = []

        for t in self.cfg.log_sample_timesteps:  # sample with to different timesteps
            (
                grad_raw,
                noise_pred,
                noise_pred_phi,
                noise,
                noisy_latents
            ) = self.compute_vsd(latents, torch.tensor([t]).to(self.device)).values()

            grad = torch.nan_to_num(
                grad_raw * self.w_schedule(step)  # apply weight schedule
            )

            pred_latents_x0 = self.scheduler.step(noise_pred, t, noisy_latents).pred_original_sample
            images_x0 = decode_latents(pred_latents_x0, self.vae)  # 2. pretrained unet predicted x0 images

            grad_abs = F.interpolate(grad.abs().mean(1, keepdim=True), size=rendered_images.shape[-2:]).cpu()
            heatmap = get_heatmap(grad_abs, size=rendered_images.shape[-2:])  # 3. grad heatmap

            phi_pred_latents_x0 = self.scheduler.step(noise_pred_phi, t, noisy_latents).pred_original_sample
            images_phi_x0 = decode_latents(phi_pred_latents_x0, self.vae)  # 4. phi unet predicted x0 images

            pred_latents_x0_pseudo_gt = self.scheduler.step(noise_pred - noise_pred_phi + noise,
                                                            t,
                                                            noisy_latents).pred_original_sample
            images_x0_pseudo_gt = decode_latents(pred_latents_x0_pseudo_gt, self.vae)  # 5. predicted pseudo gt

            images_x0s.append(images_x0)
            images_phi_x0s.append(images_phi_x0)
            images_x0_pseudo_gts.append(images_x0_pseudo_gt)
            heatmaps.append(heatmap)

        def make_labeled_grid(images, labels):  # noqa
            images = torch.cat(images, dim=0)
            return add_text_label(make_grid(images, nrow=bsz), labels)

        writer.add_image(f"visualization/predict_x0",
                         make_labeled_grid(images_x0s, [f"t={t}" for t in self.cfg.log_sample_timesteps]), step)
        writer.add_image(f"visualization/predict_x0_phi",
                         make_labeled_grid(images_phi_x0s, [f"t={t}" for t in self.cfg.log_sample_timesteps]), step)
        writer.add_image(f"visualization/predict_x0_pseudo_gt",
                         make_labeled_grid(images_x0_pseudo_gts, [f"t={t}" for t in self.cfg.log_sample_timesteps]),
                         step)
        writer.add_image(f"visualization/grad",
                         make_labeled_grid(heatmaps, [f"t={t}" for t in self.cfg.log_sample_timesteps]), step)

        # eval clip score
        for clip_name in self.cfg.eval_clip_backbones:
            clip_similarity = get_clip_similarity(clip_name).to_device(self.device)
            clip_score = 0
            for img in rendered_images:
                rgb_pil = ToPILImage()(img)
                clip_score += clip_similarity.compute_text_img_similarity(rgb_pil, self.cfg.prompt)
            clip_score /= len(rendered_images)
            writer.add_scalar(f"clip_score/{clip_name}", clip_score, step)
