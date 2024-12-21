from dataclasses import dataclass

import math
import torch
import torch.nn as nn
from einops import rearrange
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from torchvision.utils import make_grid

from .base import BaseRasterizer
from .. import register
from ..utils.log import get_density_fig, fig_to_tensor
from ..utils.misc import step_check
from ..utils.stable_diffusion import encode_images
from ..utils.typings import DictConfig, Union


def inverse_sigmoid(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.log(x / (1 - x))


@register("gaussian_2d")
class Gaussian2D(BaseRasterizer):
    @dataclass
    class Config(BaseRasterizer.Config):
        name: str
        height: int
        width: int
        batch_size: int

        device: str
        num_points: int

        lr: DictConfig

        log_interval: int

        init_opacity: float = 0.1

    cfg: Config

    def __init__(self, cfg: Union[dict, DictConfig]):
        super().__init__()
        self.cfg = self.validate_config(cfg)
        self.device = torch.device(cfg.device)

        self.num_points = cfg.num_points

        fov_x = math.pi / 2.0
        self.H, self.W = cfg.height, cfg.width
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self._init_batched_gaussians()

        param_groups = [
            {
                "params": params,
                "lr": cfg.lr[name],
                "name": name,
            }
            for name, params in self.named_parameters()
        ]

        self.optimizer = torch.optim.Adam(param_groups)

        self.log_state_dict = {}  # for logging

    def _init_batched_gaussians(self):
        """Random gaussians"""
        bsz = self.cfg.batch_size

        bd = 2

        self.means = bd * (torch.rand(bsz, self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(bsz, self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(bsz, self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        ).repeat(bsz, 1, 1)

        self.opacities = (torch.ones((bsz, self.num_points, 1), device=self.device) *
                          inverse_sigmoid(self.cfg.init_opacity))

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means = nn.Parameter(self.means)
        self.scales = nn.Parameter(self.scales)
        self.quats = nn.Parameter(self.quats)
        self.rgbs = nn.Parameter(self.rgbs)
        self.opacities = nn.Parameter(self.opacities)

        self.viewmat.requires_grad = False

    def get_latents(self, vae=None):
        assert vae is not None, "VAE is required to encode latents"
        return encode_images(self.get_images(vae), vae)

    def get_images(self, vae=None):
        """

        Args:
            vae:

        Returns: image tensor (B, C, H, W), range [0, 1]

        """
        B_SIZE = 16
        images = []

        self.log_state_dict = {
            "xys": [],
        }

        for b in range(self.cfg.batch_size):
            (
                xys,
                depths,
                radii,
                conics,
                compensation,
                num_tiles_hit,
                cov3d,
            ) = project_gaussians(
                self.means[b],
                self.scales[b],
                1,
                self.quats[b],
                self.viewmat,
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                B_SIZE,
            )
            torch.cuda.synchronize()
            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs[b]),
                torch.sigmoid(self.opacities[b]),
                self.H,
                self.W,
                B_SIZE,
                self.background,
            )[..., :3]
            torch.cuda.synchronize()

            images.append(out_img)

            self.log_state_dict["xys"].append(xys)

        return rearrange(torch.stack(images), "b h w c -> b c h w")

    def log(self, writer, step):
        """
        Log images to tensorboard
        """
        if step_check(step, self.cfg.log_interval, run_at_zero=True):
            self.get_images()  # run forward to get states

            density_fig_tensor = torch.stack([
                fig_to_tensor(get_density_fig(xy)) for xy in self.log_state_dict["xys"]
            ])  # N, C, H, W

            writer.add_image(
                "gaussians/density",
                make_grid(density_fig_tensor, nrow=self.cfg.batch_size),
                step,
            )

            writer.add_scalar("gaussians/opacity_max", torch.sigmoid(self.opacities).max(), step)
            writer.add_scalar("gaussians/opacity_min", torch.sigmoid(self.opacities).min(), step)
            writer.add_scalar("gaussians/opacity_mean", torch.sigmoid(self.opacities).mean(), step)

            writer.add_scalar("gaussians/scale_max", self.scales.max(), step)
            writer.add_scalar("gaussians/scale_min", self.scales.min(), step)
            writer.add_scalar("gaussians/scale_mean", self.scales.mean(), step)

    def step(self, _=None):
        """
        Perform optimization step
        """
        self.optimizer.step()
        self.optimizer.zero_grad()  # Clear gradients after each step
