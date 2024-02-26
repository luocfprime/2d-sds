import logging

import hydra
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithm import get_sampling_algorithm
from rasterizer import get_rasterizer
from utils.misc import seed_everything, to_primitive

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf")
def main(cfg):
    # Set seed
    seed_everything(cfg.get("seed", 42))

    # Set wandb
    if cfg.get("wandb_key", None):
        wandb.login(key=cfg.wandb_key)

    wandb.tensorboard.patch(root_logdir=str(cfg.output_path))

    wandb.init(
        project="2d-sds",
        name=cfg.uid,
        config=to_primitive(cfg),
        sync_tensorboard=True,
        # magic=True,
        save_code=True,
        group=cfg.get("group", cfg.name),
        notes=cfg.note,
        tags=cfg.get("tags", []),
    )

    writer = SummaryWriter(cfg.output_path)

    logger.info(f"output path: {cfg.output_path}")

    rasterizer = get_rasterizer(cfg.rasterizer).to(cfg.device)
    sampling_algorithm = get_sampling_algorithm(cfg.algorithm)

    # optimization loop
    for step in tqdm(range(cfg.iterations), desc="Optimization Loop"):
        sampling_algorithm.step(step, rasterizer, writer)

    logger.info("Done.")


if __name__ == "__main__":
    main()
