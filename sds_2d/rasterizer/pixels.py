from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.utils import load_image
from torchvision.transforms import ToTensor

from .. import register
from ..utils.ops import clamp
from ..utils.optimizer import get_optimizer
from ..utils.stable_diffusion import decode_latents, encode_images
from ..utils.typings import DictConfig, Union
from .base import BaseRasterizer


def randn_init(batch_size, height, width, rgb_as_latents):
    if rgb_as_latents:
        params = nn.Parameter(torch.randn(batch_size, 4, height // 8, width // 8))
        get_latent_fn = lambda params, vae: params  # noqa
        get_image_fn = lambda params, vae: decode_latents(params.data, vae)  # noqa
    else:  # Use RGB
        params = nn.Parameter(torch.randn(batch_size, 3, height, width))
        get_latent_fn = lambda params, vae: encode_images(params, vae)  # noqa
        # get_image_fn = lambda params, vae: torch.clip(params, 0, 1)  # noqa
        get_image_fn = lambda params, vae: clamp(params, 0, 1)  # noqa

    return params, get_latent_fn, get_image_fn


def constant_init(batch_size, height, width, rgb_as_latents, value):
    if rgb_as_latents:  # parameter is latent
        params = nn.Parameter(
            torch.full((batch_size, 4, height // 8, width // 8), value)
        )
        get_latent_fn = lambda params, vae: params  # noqa
        get_image_fn = lambda params, vae: decode_latents(params.data, vae)  # noqa
    else:  # Use RGB
        params = nn.Parameter(torch.full((batch_size, 3, height, width), value))
        get_latent_fn = lambda params, vae: encode_images(params, vae)  # noqa
        # get_image_fn = lambda params, vae: torch.clip(params, 0, 1)  # noqa
        get_image_fn = lambda params, vae: clamp(params, 0, 1)  # noqa

    return params, get_latent_fn, get_image_fn


def images_init(
    batch_size, height, width, rgb_as_latents, image_paths, vae_model_name_or_path
):
    images = [ToTensor()(load_image(str(p))) for p in image_paths]
    images = torch.stack(images)

    images = images.expand(batch_size, -1, -1, -1).to(torch.float32)

    assert images.shape[1] == 3, "Images must have 3 channels"
    assert (
        images.shape[2] == height and images.shape[3] == width
    ), "Images must have the same shape as the rasterizer config"

    vae = AutoencoderKL.from_pretrained(
        vae_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    )

    if rgb_as_latents:
        params = nn.Parameter(encode_images(images, vae))
        get_latent_fn = lambda params, vae: params  # noqa
        get_image_fn = lambda params, vae: decode_latents(params.data, vae)  # noqa
    else:  # Use RGB
        params = nn.Parameter(images)
        get_latent_fn = lambda params, vae: encode_images(params, vae)  # noqa
        # get_image_fn = lambda params, vae: torch.clip(params, 0, 1)  # noqa
        get_image_fn = lambda params, vae: clamp(params, 0, 1)  # noqa

    return params, get_latent_fn, get_image_fn


def constant_greyscale_init(batch_size, height, width, rgb_as_latents, value):
    assert not rgb_as_latents, "rgb_as_latents should be false when using greyscale"

    # Use RGB
    params = nn.Parameter(
        torch.full((batch_size, 1, height, width), value)
    )  # 1 for greyscale

    get_latent_fn = lambda params, vae: encode_images(
        params.expand(-1, 3, -1, -1), vae
    )  # noqa
    # get_image_fn = lambda params, vae: torch.clip(params, 0, 1).expand(-1, 3, -1, -1)  # noqa
    get_image_fn = lambda params, vae: clamp(params, 0, 1).expand(-1, 3, -1, -1)  # noqa

    return params, get_latent_fn, get_image_fn


def initialize(batch_size, height, width, rgb_as_latents, init_strategy):
    """
    Initialize the parameters of the rasterizer in-place.
    Returns:
        params, get_latent_fn(params, vae), get_image_fn(params, vae)
    """
    if init_strategy.name == "randn":
        return randn_init(batch_size, height, width, rgb_as_latents)
    elif init_strategy.name == "constant":
        return constant_init(
            batch_size, height, width, rgb_as_latents, init_strategy.args.value
        )
    elif init_strategy.name == "images":
        return images_init(
            batch_size,
            height,
            width,
            rgb_as_latents,
            init_strategy.args.image_paths,
            init_strategy.args.vae_model_name_or_path,
        )
    elif init_strategy.name == "constant_greyscale":
        return constant_greyscale_init(
            batch_size, height, width, rgb_as_latents, init_strategy.args.value
        )
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy.name}")


@register("pixels")
class Pixels(BaseRasterizer):
    @dataclass
    class Config(BaseRasterizer.Config):
        name: str

        height: int
        width: int
        batch_size: int

        dtype: str

        init_strategy: DictConfig  # {"name": str, "args": dict}

        rgb_as_latents: bool  # parameter is latent if true

        optimizer: DictConfig

    cfg: Config

    def __init__(self, cfg: Union[dict, DictConfig]):
        super().__init__()

        self.cfg = self.validate_config(cfg)

        self.params, self.get_latents_fn, self.get_images_fn = initialize(
            self.cfg.batch_size,
            self.cfg.height,
            self.cfg.width,
            self.cfg.rgb_as_latents,
            self.cfg.init_strategy,
        )

        dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[self.cfg.dtype]
        self.params = self.params.to(dtype)

        self.optimizer = get_optimizer(cfg.optimizer, [self.params])

    def get_latents(self, vae=None):
        """
        Get latents for latent diffusion model. In this case a simple wrap around to unify the interface of different rasterizers.
        Args:
            vae:

        Returns:

        """
        if not self.cfg.rgb_as_latents:
            assert vae is not None, "VAE is required to encode latents"
        return self.get_latents_fn(self.params, vae)

    def get_images(self, vae=None):
        """
        Get images from latents or directly from RGB depending on the configuration. A simple wrap around to unify the interface of different rasterizers.
        Args:
            vae:

        Returns:

        """
        if self.cfg.rgb_as_latents:
            assert vae is not None, "VAE is required to decode latents"

        return self.get_images_fn(self.params, vae)

    def step(self, _=None):
        self.optimizer.step()
        self.optimizer.zero_grad()  # Clear gradients after each step
