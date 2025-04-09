import torch
from torch.amp import custom_bwd, custom_fwd


class Clamp(torch.autograd.Function):
    """
    Clamp the input tensor, but still allow backpropagation.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


clamp = Clamp.apply
