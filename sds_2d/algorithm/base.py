from ..utils.config import ValidateConfigMixin


class BaseAlgorithm(ValidateConfigMixin):
    """
    Base class for all optimization based sampling algorithms
    """

    def step(self, step, rasterizer, writer):
        """
        Perform a single optimization step
        """
        raise NotImplementedError
