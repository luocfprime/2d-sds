import gc
import logging
import os

import hydra
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg):
    OmegaConf.resolve(cfg)  # resolve omegaconf in-place
    OmegaConf.save(cfg, f"{cfg.output_path}/config.yaml")

    if cfg.get("typecheck", False):
        from jaxtyping import install_import_hook

        install_import_hook("sds_2d", "typeguard.typechecked")

    import sds_2d  # noqa: F401
    from sds_2d import instantiate
    from sds_2d.utils.config import to_primitive
    from sds_2d.utils.misc import save_code_snapshot, seed_everything

    # Set seed
    seed_everything(cfg.get("seed", 42))

    # Set wandb
    if cfg.get("wandb_key", None):
        wandb.login(key=cfg.wandb_key)

    wandb.tensorboard.patch(root_logdir=str(cfg.output_path))

    wandb.init(config=to_primitive(cfg), **cfg.wandb)

    writer = SummaryWriter(cfg.output_path)

    os.symlink(
        wandb.run.dir, f"{cfg.output_path}/wandb_files", target_is_directory=True
    )

    if cfg.get("save_code_snapshot", False):
        save_code_snapshot(f"{cfg.output_path}/code")

    logger.info(f"output path: {cfg.output_path}")

    rasterizer = instantiate(cfg.rasterizer).to(cfg.device)
    sampling_algorithm = instantiate(cfg.algorithm)

    print(f"Starting optimization...")
    # optimization loop
    for step in tqdm(range(cfg.iterations), desc="Optimization Loop"):
        sampling_algorithm.step(step, rasterizer, writer)
        rasterizer.log(writer, step)

    logger.info("Done.")

    gc.collect()
    torch.cuda.empty_cache()

    wandb.finish()


if __name__ == "__main__":
    main()
