import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from t_schedule import get_t_schedule
from utils.log import grad_heatmap
from utils.misc import step_check
from utils.stable_diffusion import predict_noise0_diffuser_multistep, decode_latents
from utils.types import DictConfig, Union
from w_schedule import get_w_schedule
from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


class SDS(BaseAlgorithm):
    """
    Score Distillation Sampling
    """

    @dataclass
    class Config:
        name: str
        device: str
        half_inference: bool
        model_path: str
        prompt: str
        neg_prompt: str
        w_schedule_cfg: DictConfig
        t_schedule_cfg: DictConfig
        multi_steps: int
        guidance_scale: float
        log_interval: int

    def __init__(self, cfg: Union[Config, DictConfig]):
        self.cfg = cfg

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
        scheduler = DDIMScheduler.from_pretrained(cfg.model_path, subfolder="scheduler", torch_dtype=dtype)
        scheduler.set_timesteps(len(scheduler.betas))

        if cfg.half_inference:
            unet = unet.half()
            vae = vae.half()
            text_encoder = text_encoder.half()

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

        self.w_schedule = get_w_schedule(self.scheduler.betas, cfg.w_schedule_cfg)
        self.t_schedule = get_t_schedule(cfg.t_schedule_cfg, loss_weight=self.w_schedule)

    def step(self, step, rasterizer, writer):
        # 1. get latent
        latents = rasterizer.get_latents()
        bsz = latents.shape[0]
        # 2. add noise according to t schedule
        t = torch.tensor([self.t_schedule[step]]).to(self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps=t)
        # 3. sds loss
        with torch.no_grad():
            noise_pred = predict_noise0_diffuser_multistep(unet=self.unet,
                                                           noisy_latents=noisy_latents,
                                                           text_embeddings=torch.cat(
                                                               [self.uncond_text_embed.expand(bsz, -1, -1),
                                                                self.text_embed.expand(bsz, -1, -1)]),
                                                           t=t,
                                                           guidance_scale=self.cfg.guidance_scale,
                                                           scheduler=self.scheduler,
                                                           steps=self.cfg.multi_steps,
                                                           half_inference=self.cfg.half_inference)

        grad = torch.nan_to_num(
            (noise_pred - noise) * self.w_schedule[step]  # apply weight schedule
        )

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean") / latents.shape[0]

        # 4. update
        loss.backward()
        rasterizer.step()

        # 5. log
        writer.add_scalar("loss/main", loss.item(), step)
        writer.add_scalar("grad/max", grad.abs().max().item(), step)
        writer.add_scalar("grad/mean", grad.abs().mean().item(), step)
        writer.add_scalar("grad/min", grad.abs().min().item(), step)
        writer.add_scalar("t_schedule", self.t_schedule[step], step)
        writer.add_scalar("w_schedule", self.w_schedule[step], step)

        if step_check(step, self.cfg.log_interval, run_at_zero=True):
            with torch.no_grad():
                # log images
                images = decode_latents(latents, self.vae)  # 1. original rendered images

                pred_latents_x0 = self.scheduler.step(noise_pred, t, latents).pred_original_sample
                images_x0 = decode_latents(pred_latents_x0, self.vae)  # 2. predicted x0 images

                grad_abs = F.interpolate(grad.abs().mean(1, keepdim=True), size=images.shape[-2:]).cpu()
                heatmap = grad_heatmap(grad_abs, size=images.shape[-2:])  # 3. grad heatmap

            writer.add_images("visualization/rendered", images, step)
            writer.add_images("visualization/predict_x0", images_x0, step)
            writer.add_images("visualization/grad", heatmap, step, dataformats="CHW")
