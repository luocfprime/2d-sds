import inspect
import os
import random
import shutil
import subprocess
from functools import wraps

import numpy as np
import torch

from .typings import Any, Callable, Tuple


def seed_everything(seed: int = 42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval. Step count starts from 0."""
    if step_size == 0:
        return False
    return (run_at_zero and step == 0) or (step + 1) % step_size == 0


def get_file_list():
    return [
        b.decode()
        for b in set(
            subprocess.check_output(
                'git ls-files -- ":!:load/*"', shell=True
            ).splitlines()
        )
                 | set(  # hard code, TODO: use config to exclude folders or files
            subprocess.check_output(
                "git ls-files --others --exclude-standard", shell=True
            ).splitlines()
        )
    ]


def save_code_snapshot(savedir: str):
    os.makedirs(savedir, exist_ok=True)
    for f in get_file_list():
        if not os.path.exists(f) or os.path.isdir(f):
            continue
        os.makedirs(os.path.join(savedir, os.path.dirname(f)), exist_ok=True)
        shutil.copyfile(f, os.path.join(savedir, f))
