import torch.nn as nn


class BaseRasterizer(nn.Module):
    def get_latents(self, vae=None):
        raise NotImplementedError

    def get_images(self, vae=None):
        raise NotImplementedError

    def step(self):
        """
        Perform optimization step
        """
        raise NotImplementedError
