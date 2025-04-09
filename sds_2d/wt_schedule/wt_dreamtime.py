from dataclasses import dataclass

import numpy as np
import torch
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt

from .. import register
from ..utils.typings import DictConfig
from .base import BaseWTSchedule


@register("dreamtime")
class DreamtimeWTSchedule(BaseWTSchedule):
    @dataclass
    class Config:
        name: str

        model_path: str
        iterations: int

        m: float  # range [0, len(betas)]
        s: float  # range [0, inf]

    cfg: Config

    def __init__(self, cfg: DictConfig):
        self.cfg = self.validate_config(cfg)

        scheduler = DDPMScheduler.from_pretrained(
            self.cfg.model_path, subfolder="scheduler"
        )

        T = len(scheduler.betas)
        N = self.cfg.iterations

        t = torch.arange(0, T)

        alphas_cumprod = torch.flip(
            scheduler.alphas_cumprod, dims=[0]
        )  # from T -> 0 to 0 -> T

        sig_scales = torch.sqrt(alphas_cumprod)  # ddpm scale of signal
        sigmas = torch.sqrt(1.0 - alphas_cumprod)  # ddpm std of noise

        snrs = (sig_scales / sigmas) ** 2  # signal-to-noise ratio

        # weight schedule
        m = self.cfg.m
        s = self.cfg.s

        W_d = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        W_p = torch.exp(-((t - m) ** 2) / (2 * s**2))
        Z = torch.sum(W_d * W_p)

        W_t = W_d * W_p / Z

        i = torch.arange(0, N)
        t_i = torch.argmin(
            torch.abs(
                torch.flip(torch.cumsum(torch.flip(W_t, dims=[0]), 0), dims=[0])[
                    None, :
                ]
                - i[:, None] / N
            ),
            dim=1,
        )  # broadcast: [1, T] - [N, 1] -> [N, T] -> argmin dim=1 -> [N]

        self.W_d = W_d
        self.W_p = W_p

        self.W_t = W_t
        self.t_i = t_i

    def w_schedule(self, optimization_step):
        return self.W_t[self.t_i[optimization_step]]

    def t_schedule(self, optimization_step):
        return self.t_i[optimization_step]

    def show_plot(self, n_steps):
        """
        Plot the weight and timestep schedules
        """
        fig_width, fig_height = 10, 5
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height * 3))
        steps = list(range(n_steps))
        axes[0].plot(steps, [self.w_schedule(step) for step in steps])
        axes[0].set_title("Weight Schedule")
        axes[0].set_xlabel("Optimization Step")
        axes[0].set_ylabel("Weight")

        axes[1].plot(steps, [self.t_schedule(step) for step in steps])
        axes[1].set_title("Timestep Schedule")
        axes[1].set_xlabel("Optimization Step")
        axes[1].set_ylabel("Timestep")

        # plot W_d, W_p and W_t
        timesteps = np.arange(len(self.W_d))
        axes[2].plot(timesteps, self.W_d, "--", label="W_d")
        axes[2].plot(timesteps, self.W_p, "--", label="W_p")
        axes[2].plot(timesteps, self.W_t, "-", label="W_t")
        axes[2].set_title("W_d, W_p and W_t")
        axes[2].set_xlabel("Timesteps")
        axes[2].set_ylabel("Weight")
        axes[2].legend()

        plt.show()
