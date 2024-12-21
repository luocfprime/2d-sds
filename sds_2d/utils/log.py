import matplotlib.pyplot as plt
import mpl_scatter_density  # noqa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from einops import rearrange
from matplotlib.colors import LinearSegmentedColormap
from torchvision.transforms import ToPILImage, PILToTensor

from .typings import Union, List


def get_heatmap(tensor, size=None, cmap='viridis'):
    """
    Given a batch of tensor, colorize it with a palette and return a grid of heatmaps
    Args:
        tensor: A tensor of shape "B 3 H W" or "B 1 H W" or "B H W"
        size: The target shape of the heatmap, (H, W), if None, use the tensor.shape[-2:]
        cmap: The color palette

    Returns:
        A batched tensor heatmap of the tensor of shape "B 3 H W", range [0, 1]
    """
    assert tensor.dim() in [3, 4], f"Expected 3D or 4D tensor, got {tensor.dim()}D tensor"

    if size is None:
        size = tensor.shape[-2:]

    if tensor.dim() == 3:  # B H W
        tensor = tensor.unsqueeze(1)
    elif tensor.shape[1] == 3:  # B 3 H W
        tensor = tensor.mean(1, keepdim=True)
    elif tensor.shape[1] == 1:  # B 1 H W
        pass

    B = tensor.shape[0]

    tensor = tensor.abs()  # B 1 H W
    tensor = F.interpolate(tensor.float(), size=size)  # B 1 H W
    tensor = rearrange(tensor, "B 1 H W -> 1 H (B W)").expand(3, -1, -1)  # 3 H_ W_
    tensor = tensor[0]  # H_ W_

    tensor = tensor.detach().cpu().numpy()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(tensor.shape[1] / 100, tensor.shape[0] / 100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(tensor, cmap=cmap, aspect='auto')

    fig.canvas.draw()

    # extract image from the figure
    buffer = fig.canvas.buffer_rgba()
    buffer = np.array(buffer)
    buffer = buffer[:, :, :-1]  # remove alpha channel
    buffer = torch.from_numpy(buffer) / 255.0
    buffer = rearrange(buffer, "H (B W) C -> B C H W", B=B)

    plt.close(fig)

    return buffer


def fig_to_numpy(fig, close=True):
    """
    Convert a matplotlib figure to numpy array
    Args:
        fig: matplotlib.figure.Figure
        close: close the figure after conversion
    Returns:
        A numpy array of shape "H W C", range [0, 255]
    """
    fig.canvas.draw()

    # extract image from the figure
    buffer = fig.canvas.buffer_rgba()
    buffer = np.array(buffer)
    buffer = buffer[:, :, :-1]  # remove alpha channel

    if close:
        plt.close(fig)

    return buffer


def fig_to_tensor(fig, close=True):
    """
    Convert a matplotlib figure to torch tensor
    Args:
        fig: matplotlib.figure.Figure
        close: close the figure after conversion
    Returns:
        A tensor of shape "C H W", dtype float32, range [0, 1]
    """
    buffer = fig_to_numpy(fig, close=close)
    buffer = torch.from_numpy(buffer) / 255.0
    buffer = rearrange(buffer, "H W C -> C H W")

    return buffer


def get_density_fig(xy):
    """
    Given a set of 2D points, return a density plot
    Args:
        xy: [N, 2] tensor

    Returns:
        fig: matplotlib.figure.Figure
    """
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    xy = xy.detach().cpu().numpy().round().astype(int)
    density = ax.scatter_density(xy[:, 0], xy[:, 1], cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    return fig


def show_image(img: Union[torch.Tensor, np.ndarray, Image.Image]):
    """
    Display a image via matplotlib
    for tensor, the shape should be "3 H W", dtype should be float32, range [0, 1], RGB
    for np.ndarray, the shape should be "H W C", dtype should be uint8, range [0, 255], RGB
    """
    if isinstance(img, torch.Tensor) or isinstance(img, np.ndarray):
        img = ToPILImage()(img)
    elif not isinstance(img, Image.Image):
        raise ValueError(f"Expected torch.Tensor, np.ndarray or PIL.Image, got {type(img)}")

    # show image without frame
    fig = plt.figure(frameon=False)
    fig.set_size_inches(img.size[0] / 100, img.size[1] / 100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img)
    plt.show()


def add_text_label(img: Union[torch.Tensor, np.ndarray, Image.Image], texts: List[str]):
    """
    Add text labels to single image, len(texts) is number of rows
    for tensor, the shape should be "3 H W", dtype should be float32, range [0, 1], RGB
    for np.ndarray, the shape should be "H W C", dtype should be uint8, range [0, 255], RGB
    """
    input_type = type(img)
    if isinstance(img, torch.Tensor):
        img = ToPILImage()(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        raise ValueError(f"Expected torch.Tensor, np.ndarray or PIL.Image, got {type(img)}")

    # add text label
    draw = ImageDraw.Draw(img)
    black, white = (0, 0, 0), (255, 255, 255)
    for i, text in enumerate(texts):
        draw.text((2, (img.size[1] // len(texts)) * i + 1), f"{text}", white)
        draw.text((0, (img.size[1] // len(texts)) * i + 1), f"{text}", white)
        draw.text((2, (img.size[1] // len(texts)) * i - 1), f"{text}", white)
        draw.text((0, (img.size[1] // len(texts)) * i - 1), f"{text}", white)
        draw.text((1, (img.size[1] // len(texts)) * i), f"{text}", black)

    if input_type == torch.Tensor:
        img = PILToTensor()(img)
    elif input_type == np.ndarray:
        img = np.asarray(img)

    return img

# if __name__ == "__main__":
#     grad_tensor = torch.randn(9, 3, 64, 64)
#     heatmap = grad_heatmap(grad_tensor, size=(64, 64))  # C H W
#     print(heatmap.shape)
#
#     img_np = heatmap.numpy()
#
#     plt.imshow(img_np.transpose(1, 2, 0))  # C H W -> H W C
#     plt.show()
