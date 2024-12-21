from pathlib import Path

import torch
from diffusers.utils import load_image
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid

from sds_2d.rasterizer import get_rasterizer
from sds_2d.utils.log import get_density_fig, fig_to_tensor, show_image


if __name__ == "__main__":
    device = "cuda:0"

    gt_images_paths = list(Path("../imgs/").glob("*.jpg")) + list(Path("../imgs/").glob("*.png"))
    gt_images = [ToTensor()(load_image(str(p))) for p in gt_images_paths]
    gt_images = torch.stack(gt_images).to(device)

    cfg = OmegaConf.create(
        {
            "name": "gaussian_2d",
            "height": 512,
            "width": 512,
            "batch_size": gt_images.shape[0],
            "device": device,
            "num_points": 10000,
            "log_interval": 10,
            # "optimizer": {
            #     "name": "Adam",
            #     "opt_args": {
            #         "lr": 0.01
            #     }
            # }
            "lr": {
                "means": 0.01,
                "scales": 0.01,
                "quats": 0.01,
                "rgbs": 0.1,
                "opacities": 0.1,
            }
        }
    )

    rasterizer = get_rasterizer(cfg).to(device)

    densities = []
    for i in range(100):  # iterations
        rendered_images = rasterizer.get_images()
        loss = 0.5 * ((gt_images.detach() - rendered_images) ** 2).mean()
        loss.backward()
        rasterizer.step()
        print(f"Step {i}: Loss {loss.item()}")

        # log densities
        density_fig_tensor = torch.stack([
            fig_to_tensor(get_density_fig(xy)) for xy in rasterizer.log_state_dict["xys"]
        ])  # N, C, H, W
        densities.append(make_grid(density_fig_tensor, nrow=rasterizer.cfg.batch_size))

    # save the final images
    Path("./renders").mkdir(exist_ok=True, parents=True)
    rendered_images = rasterizer.get_images()
    for i, img in enumerate(rendered_images):
        img = ToPILImage()(img)
        img.save(f"./renders/output_{i}.png")

    for i, img in enumerate(densities):
        img = ToPILImage()(img)
        img.save(f"./renders/density_{i}.png")

    show_image(make_grid(torch.cat([gt_images, rendered_images], dim=0), nrow=cfg.batch_size))

    print("Done.")
