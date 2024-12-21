from pathlib import Path

import torch
from diffusers import AutoencoderKL
from diffusers.utils import load_image
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from sds_2d.utils.stable_diffusion import encode_images
from sds_2d.rasterizer import get_rasterizer
from sds_2d.utils.log import show_image, add_text_label
from sds_2d.utils.misc import step_check

if __name__ == "__main__":
    device = "cuda:0"
    torch.set_default_device(device)

    gt_images_paths = [Path("./imgs/pseudo_gt.png")] #list(Path("../imgs/").glob("*.jpg")) + list(Path("../imgs/").glob("*.png"))
    gt_images = [ToTensor()(load_image(str(p))) for p in gt_images_paths]
    gt_images = torch.stack(gt_images).to(device)

    cfg = OmegaConf.create(
        {
            "name": "pixels",
            "height": 512,
            "width": 512,
            "batch_size": len(gt_images_paths),
            "rgb_as_latents": False,  # check this

            "init_strategy": {
                "name": "constant",
                "args": {
                    "value": 0.5
                }
            },

            # "init_strategy": {
            #     "name": "constant_greyscale",
            #     "args": {
            #         "value": 0.5
            #     }
            # },

            "optimizer": {
                "name": "Adam",
                "opt_args": {
                    "lr": 0.05
                }
            }
        }
    )

    rasterizer = get_rasterizer(cfg).to(device)

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                        subfolder="vae",
                                        torch_dtype=torch.float16)
    vae = vae.to(device)

    lambda_latent = 0.0
    lambda_img = 1.0 - lambda_latent

    rendered_images_process = []
    log_interval = 10
    iterations = 50  # 500

    loss_img_list = []
    loss_latent_list = []

    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(iterations):  # iterations
            rendered_images = rasterizer.get_images(vae=vae)
            latents = rasterizer.get_latents(vae=vae)

            loss_img = 0.5 * ((gt_images.detach() - rendered_images) ** 2).mean()  # image matching
            loss_latent = 0.5 * ((encode_images(gt_images, vae).detach() - latents) ** 2).mean()  # latent matching would produce artifacts

            loss = lambda_img * loss_img + lambda_latent * loss_latent
            loss.backward()
            rasterizer.step()

            loss_img_list.append(loss_img.item())
            loss_latent_list.append(loss_latent.item())

            print(f"Step {i}: Loss {loss.item()}, Loss_img {loss_img.item()}, Loss_latent {loss_latent.item()}")
            if step_check(i, log_interval, run_at_zero=True):
                rendered_images_process.append(rendered_images)

    rendered_images_process_tensor = rearrange(torch.stack(rendered_images_process), 'it b c h w -> c (b h) (it w)')
    gt_images = rearrange(gt_images, 'b c h w -> c (b h) w')
    show_image(
        add_text_label(torch.cat([gt_images, rendered_images_process_tensor], dim=2),
                       ["first column: GT; x-axis: iterations"]),
    )  # cat along width

    axes = plt.figure().subplots(1, 2)
    axes[0].plot(loss_img_list, label="Loss/img")
    axes[0].legend()
    axes[1].plot(loss_latent_list, label="Loss/latent")
    axes[1].legend()
    plt.show()

    print("Done.")
