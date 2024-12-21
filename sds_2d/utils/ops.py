import torch
from torch.cuda.amp import custom_bwd, custom_fwd


class Clamp(torch.autograd.Function):
    """
    Clamp the input tensor, but still allow backpropagation.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


clamp = Clamp.apply
