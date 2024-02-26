import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid


def grad_heatmap(grad_tensor, size, padding=2):
    """
    Given a gradient tensor, return a heatmap of the gradient.
    Args:
        grad_tensor: A gradient tensor of shape "B 3 H W"
        size: The size of the heatmap
        padding: Padding of the grid

    Returns:
        A heatmap of the gradient tensor of shape "3 H (B W)", range [0, 1]
    """
    grad_abs = grad_tensor.abs().mean(1, keepdim=True)  # B 1 H W
    grad_abs = F.interpolate(grad_abs, size=size)  # B 1 H W
    grad_abs = make_grid(grad_abs, padding=padding)  # 3 H_ W_
    grad_abs = grad_abs[0]  # H_ W_

    grad_abs = grad_abs.detach().cpu().numpy()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(grad_abs.shape[1] / 100, grad_abs.shape[0] / 100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(grad_abs, cmap='viridis', aspect='auto')

    fig.canvas.draw()

    # extract image from the figure
    buffer = fig.canvas.buffer_rgba()
    buffer = np.array(buffer)
    buffer = buffer[:, :, :-1]  # remove alpha channel
    buffer = rearrange(buffer, 'h w c -> c h w')
    buffer = torch.from_numpy(buffer) / 255.0

    return buffer

# if __name__ == "__main__":
#     grad_tensor = torch.randn(9, 3, 64, 64)
#     heatmap = grad_heatmap(grad_tensor, size=(64, 64))  # C H W
#     print(heatmap.shape)
#
#     img_np = heatmap.numpy()
#
#     plt.imshow(img_np.transpose(1, 2, 0))  # C H W -> H W C
#     plt.show()
