import random

import numpy as np
import torch
from omegaconf import OmegaConf


def seed_everything(seed: int = 42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_primitive(cfg, resolve=True):
    # convert omegaconf to primitive types, avoid errors in calls of type(cfg) and isinstance(cfg, ...)
    if (
            isinstance(cfg, float)
            or isinstance(cfg, int)
            or isinstance(cfg, str)
            or isinstance(cfg, list)
            or isinstance(cfg, dict)
    ):
        return cfg
    return OmegaConf.to_container(cfg, resolve=resolve)


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval. credit: nerfstudio"""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0
