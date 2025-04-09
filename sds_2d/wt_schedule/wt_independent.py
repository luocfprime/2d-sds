from dataclasses import dataclass
from functools import partial
from warnings import warn

import numpy as np
import torch
from diffusers import DDPMScheduler

from .. import register
from ..utils.typings import DictConfig
from .base import BaseWTSchedule


# Some simple implementations of weight and timestep schedules
# implementation starts here ------------------------------------------------
def linear_schedule(start, end, n_steps: int, dtype):
    """Return linearly spaced integers from `start` (inclusive) to `end` (inclusive)."""
    steps = np.linspace(start, end, n_steps, dtype=dtype, endpoint=True)
    return lambda x: steps[x]


def constant_schedule(value):
    return lambda x: value


def discrete_uniform_schedule(low, high):
    """Return random integers from `low` (inclusive) to `high` (exclusive)."""
    return lambda x: np.random.randint(low, high)


def random_decay_schedule(
    low_start, high_start, low_end, high_end, n_steps, decay_func: str = "linear"
):
    """
    lower bounds (low_*) are inclusive, upper bounds are exclusive
    """
    assert decay_func in [
        "linear",
        "exp",
    ], f"decay_func currently not supported: {decay_func}"
    if decay_func == "linear":
        lower_bound = np.linspace(low_start, low_end, n_steps, dtype=int)
        upper_bound = np.linspace(high_start, high_end, n_steps, dtype=int)
    else:
        lower_bound = np.floor(
            np.exp(np.linspace(np.log(low_start), np.log(low_end), n_steps))
        )
        upper_bound = np.ceil(
            np.exp(np.linspace(np.log(high_start), np.log(high_end), n_steps))
        )

    return lambda x: np.random.randint(lower_bound[x], upper_bound[x])


def polynomial_schedule(coefs, n_steps, dtype):
    """Return a polynomial schedule."""
    steps = np.polyval(coefs, np.linspace(0, 1, n_steps)).astype(dtype)
    return lambda x: steps[x]


def hifa_t_schedule(t_max, t_min, n_steps):
    """Return a HIFA timestep schedule."""
    return lambda x: int(t_max - (t_max - t_min) * np.sqrt(x / n_steps))


def piecewise_linear_schedule(pts, dtype=int):
    """
    Return a piecewise linear schedule.
    pts: list of tuples [(x1, y1), (x2, y2), ...]
    """
    x, y = zip(*pts)
    if not all(
        x[i] <= x[i + 1] for i in range(len(x) - 1)
    ):  # if x not in ascending order, warn it
        warn("x values are not in ascending order")
    x = np.array(x)
    y = np.array(y)
    return lambda t: np.interp(t, x, y).astype(dtype)


def stepping_schedule(pts):
    """
    Return a stepping schedule.
    pts: list of tuples [(x1, y1), (x2, y2), ...]
    """
    x, y = zip(*pts)
    if not all(
        x[i] <= x[i + 1] for i in range(len(x) - 1)
    ):  # if x not in ascending order, warn it
        warn("x values are not in ascending order")
    x = np.array(x)
    y = np.array(y)
    return lambda t: y[np.searchsorted(x, t, side="right") - 1]


class diffusion_based_w_schedule:  # noqa
    """Schedules related to diffusion process"""

    def __init__(self, model_path, method_name):
        self.method_name = method_name

        scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        self.betas = scheduler.betas
        self.alphas = scheduler.alphas
        self.alphas_cumprod = (
            scheduler.alphas_cumprod
        )  # alphas_cumprod is 'alphas' in threestudio

        self.sig_scales = torch.sqrt(self.alphas_cumprod)  # ddpm scale of signal
        self.sigmas = torch.sqrt(1.0 - self.alphas_cumprod)  # ddpm std of noise

    def __call__(self, x):
        if self.method_name == "dreamfusion":
            return 1.0 - self.alphas_cumprod[x]  # \sigma_t^2
        elif self.method_name == "fantasia3d":
            return self.alphas_cumprod[x] ** 0.5 * (
                1 - self.alphas_cumprod[x]
            )  # \sigma_t^2 * \sqrt(1 - \sigma_t^2)
        else:
            raise NotImplementedError(f"method_name {self.method_name} not implemented")


# implementation ends here ------------------------------------------------


@register("independent")
class IndependentWTSchedule(BaseWTSchedule):
    """
    Independent weight and timestep schedule,
    i.e. weight schedule and timestep schedule are independent of each other
    """

    @dataclass
    class Config(BaseWTSchedule.Config):
        name: str

        n_steps: int
        w_schedule_cfg: DictConfig
        t_schedule_cfg: DictConfig

    cfg: Config

    independent_t_schedule_registry = {
        "uniform": discrete_uniform_schedule,
        "linear": partial(linear_schedule, dtype=int),
        "constant": constant_schedule,
        "random_decay": random_decay_schedule,
        "polynomial": partial(polynomial_schedule, dtype=int),
        "hifa": hifa_t_schedule,
        "piecewise_linear": partial(piecewise_linear_schedule, dtype=int),
        "stepping": stepping_schedule,
    }

    independent_w_schedule_registry = {
        "linear": partial(linear_schedule, dtype=float),
        "constant": constant_schedule,
        "polynomial": partial(polynomial_schedule, dtype=float),
        "dreamfusion": partial(diffusion_based_w_schedule, method_name="dreamfusion"),
        "fantasia3d": partial(diffusion_based_w_schedule, method_name="fantasia3d"),
        "piecewise_linear": partial(piecewise_linear_schedule, dtype=float),
        "stepping": stepping_schedule,
    }

    def __init__(self, cfg):
        """
        Args:
            cfg: config containing the weight and timestep schedule configurations
        """
        self.cfg = self.validate_config(cfg)

        w_schedule_args = self.cfg.w_schedule_cfg.get("args", {})
        w_schedule_fn = self.independent_w_schedule_registry[
            self.cfg.w_schedule_cfg.name
        ](**w_schedule_args)

        t_schedule_args = self.cfg.t_schedule_cfg.get("args", {})
        t_schedule_fn = self.independent_t_schedule_registry[
            self.cfg.t_schedule_cfg.name
        ](**t_schedule_args)

        optimization_steps = list(range(self.cfg.n_steps))

        self._t_schedule = [t_schedule_fn(x) for x in optimization_steps]
        self._w_schedule = [w_schedule_fn(t) for t in self._t_schedule]

    def t_schedule(self, optimization_step):
        return self._t_schedule[optimization_step]

    def w_schedule(self, optimization_step):
        return self._w_schedule[optimization_step]
