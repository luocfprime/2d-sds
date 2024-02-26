from dataclasses import dataclass

import numpy as np
import torch

from utils.types import DictConfig, Union


@dataclass
class WeightScheduleConfig:
    name: str
    sigma_y: float  # optional


def get_w_schedule(betas, cfg: Union[WeightScheduleConfig, DictConfig]) -> list:
    num_train_timesteps = len(betas)
    betas = torch.tensor(betas) if not torch.is_tensor(betas) else betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)  # equivalent noise sigma on image
    sigma_ks = []
    SNRs = []
    rhos = []
    m1 = 800
    m2 = 500
    s1 = 300
    s2 = 100
    for i in range(num_train_timesteps):
        sigma_ks.append(reduced_alpha_cumprod[i])
        SNRs.append(1 / reduced_alpha_cumprod[i])
        if cfg.name == 'rhos':
            rhos.append(1. * (cfg.sigma_y ** 2) / (sigma_ks[i] ** 2))

    def loss_weight(t):
        if cfg.name == None or cfg.name == 'none':
            return 1
        elif 'SNR' in cfg.name:
            ## ref: https://arxiv.org/abs/2305.04391
            if cfg.name == 'SNR':
                return 1 / SNRs[t]
            elif cfg.name == 'SNR_sqrt':
                return torch.sqrt(1 / SNRs[t])
            elif cfg.name == 'SNR_square':
                return (1 / SNRs[t]) ** 2
            elif cfg.name == 'SNR_log1p':
                return torch.log(1 + 1 / SNRs[t])
        elif cfg.name == 'rhos':
            return 1 / rhos[t]
        elif 'alpha' in cfg.name:
            if cfg.name == 'sqrt_alphas_cumprod':
                return sqrt_alphas_cumprod[t]
            elif cfg.name == '1m_alphas_cumprod':
                return sqrt_1m_alphas_cumprod[t] ** 2
            elif cfg.name == 'alphas_cumprod':
                return alphas_cumprod[t]
            elif cfg.name == 'sqrt_alphas_1m_alphas_cumprod':
                return sqrt_alphas_cumprod[t] * sqrt_1m_alphas_cumprod[t]
        elif 'dreamtime' in cfg.name:
            if t > m1:
                return np.exp(-(t - m1) ** 2 / (2 * s1 ** 2))
            elif t >= m2:
                return 1
            else:
                return np.exp(-(t - m2) ** 2 / (2 * s2 ** 2))
        else:
            raise NotImplementedError

    weights = []
    for i in range(num_train_timesteps):
        weights.append(loss_weight(i))
    return weights
