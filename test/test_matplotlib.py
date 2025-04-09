import time

import numpy as np
import torch
from torchvision.utils import make_grid

from sds_2d.utils.log import add_text_label, get_heatmap, show_image

if __name__ == "__main__":
    # x = torch.FloatTensor(3, 256, 256).uniform_(0, 1)  # 1. test with torch.Tensor, CHW
    # x[0, :, :] = 0  # set R channel to 0
    # # x = np.random.randint(0, 256, (256, 256, 3)).astype(np.uint8)  # 2. test with np.ndarray, HWC
    # # x[:, :, 0] = 0  # set R channel to 0
    # show_image(add_text_label(x, ["test1", "test2"]))

    heatmaps = []
    for i in range(50):
        grad_tensor = torch.randn(50, 3, 64, 64)
        heatmap = get_heatmap(grad_tensor, size=(64, 64))  # C H W
        show_image(make_grid(heatmap, nrow=3))  # C H W
        time.sleep(2)
        heatmaps.append(heatmap)

        print(len(heatmaps))
