import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omegaconf import OmegaConf

from sds_2d.wt_schedule import get_wt_schedule

if __name__ == "__main__":
    cfg = OmegaConf.create(
        {
            "name": "dreamtime",
            "model_path": "stabilityai/stable-diffusion-2-1-base",
            "iterations": 500,  # 5000
            "m": 800,
            "s": 300,
        }
    )

    wt_schedule = get_wt_schedule(cfg)
    wt_schedule.show_plot(cfg.iterations)
