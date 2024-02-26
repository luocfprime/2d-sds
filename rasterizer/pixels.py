from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.optimizer import get_optimizer
from utils.stable_diffusion import decode_latents
from utils.types import DictConfig, Union
from .base import BaseRasterizer


class Pixels(BaseRasterizer):
    @dataclass
    class Config:
        name: str
        height: int
        width: int
        batch_size: int

        rgb_as_latents: bool

        optimizer: DictConfig

    def __init__(self, cfg: Union[Config, DictConfig], *args, **kwargs):
        super().__init__()

        self.cfg = cfg

        if cfg.rgb_as_latents:
            self.params = nn.Parameter(torch.randn(cfg.batch_size, 4, cfg.height // 8, cfg.width // 8))
        else:  # Use RGB
            self.params = nn.Parameter(torch.randn(cfg.batch_size, 3, cfg.height, cfg.width))

        self.optimizer = get_optimizer(cfg.optimizer, [self.params])

    def get_latents(self, vae=None):
        """
        Get latents for latent diffusion model. In this case a simple wrap around to unify the interface of different rasterizers.
        Args:
            vae:

        Returns:

        """
        if self.cfg.rgb_as_latents:
            assert vae is None, "VAE should not be used when rgb_as_latents is True"
            return self.params
        else:
            assert vae is not None, "VAE is required to encode latents"
            return vae.encode(self.params)

    def get_images(self, vae=None):
        """
        Get images from latents or directly from RGB depending on the configuration. A simple wrap around to unify the interface of different rasterizers.
        Args:
            vae:

        Returns:

        """
        if self.cfg.rgb_as_latents:
            return self.params
        else:
            assert vae is not None, "VAE is required to decode latents"
            return decode_latents(self.params.data, vae)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()  # Clear gradients after each step
