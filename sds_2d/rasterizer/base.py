import torch.nn as nn
from ..utils.config import ValidateConfigMixin


class BaseRasterizer(nn.Module, ValidateConfigMixin):
    def get_latents(self, vae=None):
        raise NotImplementedError

    def get_images(self, vae=None):
        raise NotImplementedError

    def step(self, global_step):
        """
        Perform optimization step
        """
        raise NotImplementedError

    def log(self, writer, step):
        pass