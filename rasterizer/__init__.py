from .diffsvg import DiffSVG
from .pixels import Pixels

rasterizer_registry = {
    "pixels": Pixels,
    "diffsvg": DiffSVG,
}


def get_rasterizer(rasterizer_cfg):
    if rasterizer_cfg.name not in rasterizer_registry:
        raise NotImplementedError(f"Rasterizer {rasterizer_cfg.name} not implemented")
    return rasterizer_registry[rasterizer_cfg.name](rasterizer_cfg)
