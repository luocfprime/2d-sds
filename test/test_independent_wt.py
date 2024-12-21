import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omegaconf import OmegaConf

from sds_2d.wt_schedule import get_wt_schedule

if __name__ == "__main__":
    cfg = OmegaConf.create(
        # {
        #     "name": "independent",
        #     "n_steps": 1000,
        #     "w_schedule_cfg": {
        #         "name": "linear",
        #         "args": {
        #             "start": 0.0,
        #             "end": 1.0,
        #             "n_steps": 1000
        #         }
        #     },
        #     "t_schedule_cfg": {
        #         "name": "random_decay",
        #         "args": {
        #             "low_start": 0,
        #             "high_start": 1000,
        #             "low_end": 0,
        #             "high_end": 10,
        #             "n_steps": 1000,
        #             "decay_func": "linear"
        #         }
        #     }
        # }
        {
            "name": "independent",
            "n_steps": 500,
            "w_schedule_cfg": {
                "name": "dreamfusion",
                "args": {
                    "model_path": "stabilityai/stable-diffusion-2-1-base",
                }
            },
            "t_schedule_cfg": {
                "name": "uniform",
                "args": {
                    "low": 0,
                    "high": 1000
                }
            }
        }
    )

    wt_schedule = get_wt_schedule(cfg)
    wt_schedule.show_plot(cfg.n_steps)
