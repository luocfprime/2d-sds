from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from utils.types import Union, DictConfig


@dataclass
class TScheduleConfig:
    name: str
    num_train_timesteps: int  # Number of time steps for optimization
    num_steps: int  # Number of steps for random sampling (i.e. max diffusion steps)
    t_start: int
    t_end: int
    w_schedule_name: str  # optional


def get_t_schedule(cfg: Union[TScheduleConfig, DictConfig], loss_weight=None) -> list:
    # Create a list of time steps from 0 to num_train_timesteps
    ts = list(range(cfg.num_train_timesteps))
    # set ts to U[0.02,0.98] as least
    assert (cfg.t_start >= 20) and (cfg.t_end <= 980)
    ts = ts[cfg.t_start:cfg.t_end]

    # If the scheduling strategy is 'random', choose args.num_steps random time steps without replacement
    if cfg.name == 'random':
        chosen_ts = np.random.choice(ts, cfg.num_steps, replace=True)

    # If the scheduling strategy is 'random_down', first exclude the first 30 and last 10 time steps
    # then choose a random time step from an interval that shrinks as step increases
    elif 'random_down' in cfg.name:
        interval_ratio = int(cfg.name[11:]) if len(cfg.name[11:]) > 0 else 5
        interval_ratio *= 0.1
        chosen_ts = [np.random.choice(
            ts[max(0, int((cfg.num_steps - step - interval_ratio * cfg.num_steps) / cfg.num_steps * len(ts))): \
               min(len(ts), int((cfg.num_steps - step + interval_ratio * cfg.num_steps) / cfg.num_steps * len(ts)))],
            1, replace=True).astype(int)[0] \
                     for step in range(cfg.num_steps)]

    # If the scheduling strategy is 'fixed', parse the fixed time step from the string and repeat it args.num_steps times
    elif 'fixed' in cfg.name:
        fixed_t = int(cfg.name[5:])
        chosen_ts = [fixed_t for _ in range(cfg.num_steps)]

    # If the scheduling strategy is 'descend', parse the start time step from the string (or default to 1000)
    # then create a list of descending time steps from the start to 0, with length args.num_steps
    elif 'descend' in cfg.name:
        if 'quad' in cfg.name:  # no significant improvement
            descend_from = int(cfg.name[12:]) if len(cfg.name[7:]) > 0 else len(ts)
            chosen_ts = np.square(np.linspace(descend_from ** 0.5, 1, cfg.num_steps))
            chosen_ts = chosen_ts.astype(int).tolist()
        else:
            descend_from = int(cfg.name[7:]) if len(cfg.name[7:]) > 0 else len(ts)
            chosen_ts = np.linspace(descend_from - 1, 1, cfg.num_steps, endpoint=True)
            chosen_ts = chosen_ts.astype(int).tolist()

    # If the scheduling strategy is 't_stages', the total number of time steps are divided into several stages.
    # In each stage, a decreasing portion of the total time steps is considered for selection.
    # For each stage, time steps are randomly selected with replacement from the respective portion.
    # The final list of chosen time steps is a concatenation of the time steps selected in all stages.
    # Note: The total number of time steps should be evenly divisible by the number of stages.
    elif 't_stages' in cfg.name:
        # Parse the number of stages from the scheduling strategy string
        num_stages = int(cfg.name[8:]) if len(cfg.name[8:]) > 0 else 2
        chosen_ts = []
        for i in range(num_stages):
            # Define the portion of ts to be considered in this stage
            portion = ts[:int((num_stages - i) * len(ts) // num_stages)]
            selected_ts = np.random.choice(portion, cfg.num_steps // num_stages, replace=True).tolist()
            chosen_ts += selected_ts

    elif 'dreamtime' in cfg.name:
        # time schedule in dreamtime https://arxiv.org/abs//2306.12422
        assert 'dreamtime' in cfg.w_schedule_name, f"dreamtime t_scheduler requires w_schedule_name to be dreamtime"
        loss_weight_sum = np.sum(loss_weight)
        p = [wt / loss_weight_sum for wt in loss_weight]
        N = cfg.num_steps

        def t_i(t, i, p):
            t = int(max(0, min(len(p) - 1, t)))
            return abs(sum(p[t:]) - i / N)

        chosen_ts = []
        for i in range(N):
            # Initial guess for t
            t0 = 999
            selected_t = minimize(t_i, t0, args=(i, p), method='Nelder-Mead')
            selected_t = max(0, int(selected_t.x))
            chosen_ts.append(selected_t)
    else:
        raise ValueError(f"Unknown scheduling strategy: {cfg.name}")

    # Return the list of chosen time steps
    return chosen_ts
