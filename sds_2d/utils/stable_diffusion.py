from contextlib import contextmanager

import torch
from diffusers import AutoencoderKL

from .ops import clamp
from .typings import Optional


@contextmanager
def scheduler_inference_steps(scheduler, num_inference_steps, device):
    """
    A context manager to temporarily change the number of inference steps in the scheduler.
    Args:
        scheduler: diffusers scheduler
        num_inference_steps (`int`):
            the number of diffusion steps used when generating samples with a pre-trained model.
        device (`str` or `torch.device`, optional):
            the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

    Returns:
        scheduler with updated number of inference steps
    """
    old_num_inference_steps = scheduler.num_inference_steps
    scheduler.set_timesteps(num_inference_steps, device)

    yield scheduler

    if (
        old_num_inference_steps is not None
    ):  # restore the original number of inference steps
        scheduler.set_timesteps(old_num_inference_steps, device)
    else:
        scheduler.num_inference_steps = None


def encode_images(images: torch.Tensor, vae: AutoencoderKL):
    """
    Encodes images using the VAE.
    Args:
        images: range [0, 1]
        vae: VAE model

    Returns: latents

    """
    images = images.to(vae.device)
    images = images * 2.0 - 1.0
    return vae.config.scaling_factor * vae.encode(images).latent_dist.sample()


def decode_latents(latents, vae: AutoencoderKL):
    """
    Decodes latents using the VAE.

    Returns: images, range [0, 1]
    """
    latents = latents.to(vae.device)
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    # return torch.clamp((images + 1.0) / 2.0, 0, 1)
    return clamp((images + 1.0) / 2.0, 0, 1)


def predict_noise(
    unet,
    noisy_latents,
    text_embeddings,
    uncond_text_embeddings,
    t,
    *,
    scheduler=None,
    guidance_scale=7.5,
):
    """
    Predicts the epsilon noise for a given set of latents and text embeddings.
    Args:
        unet:
        noisy_latents: (bsz, 4, h, w) tensor
        text_embeddings: conditional text embeddings, (bsz, text_len, dim) tensor
        uncond_text_embeddings: unconditional text embeddings or negative text embeddings, (bsz, text_len, dim) tensor
        t: timestep, (bsz,) tensor
        scheduler: diffusers scheduler
        guidance_scale: float, default 7.5

    Returns:
        noise_pred (bsz, 4, h, w) tensor
    """
    encoder_hidden_states = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)

    if (
        scheduler is not None
    ):  # if scheduler is provided, scale the input (if necessary)
        noisy_latents = scheduler.scale_model_input(noisy_latents, t)

    noise_pred = unet(
        noisy_latents.repeat(
            2, 1, 1, 1
        ),  # (2*bsz, 4, h, w), 2*bsz for classifier-free guidance
        t.repeat(2),  # (2*bsz,)
        encoder_hidden_states=encoder_hidden_states,
    ).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

    # perform guidance
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return noise_pred


def predict_noise_no_cfg(
    unet,
    noisy_latents,
    text_embeddings,
    t,
    *,
    scheduler=None,
):
    """
    Predicts the epsilon noise for a given set of latents and text embeddings.
    Args:
        unet:
        noisy_latents: (bsz, 4, h, w) tensor
        text_embeddings: conditional text embeddings, (bsz, text_len, dim) tensor
        t: timestep, (bsz,) tensor
        scheduler: diffusers scheduler

    Returns:
        noise_pred (bsz, 4, h, w) tensor
    """
    if (
        scheduler is not None
    ):  # if scheduler is provided, scale the input (if necessary)
        noisy_latents = scheduler.scale_model_input(noisy_latents, t)

    noise_pred = unet(
        noisy_latents,  # (bsz, 4, h, w)
        t,  # (bsz,)
        encoder_hidden_states=text_embeddings,
    ).sample

    return noise_pred


def solve_diffusion_equation(
    unet,
    noisy_latents,  # x_t
    text_embeddings,  # c_0
    uncond_text_embeddings,  # c_1
    timesteps,  # [t_n, t_n-1, ..., t_0]
    scheduler,
    guidance_scale=7.5,
):
    """
    Solves the diffusion equation. From timesteps[0] to timesteps[-1], this function simulates the multi-step denoising
    Args:
        unet:
        noisy_latents:
        text_embeddings:
        uncond_text_embeddings:
        timesteps: IntTensor, shape (inference_steps,)
        scheduler:
        guidance_scale:

    Returns:
        sample_prediction
    """
    # check if timesteps matches the scheduler, i.e. items in timesteps are in scheduler.timesteps
    assert all(
        [t in scheduler.timesteps for t in timesteps]
    ), f"timesteps must be a subset of scheduler.timesteps, got {timesteps} and {scheduler.timesteps}"
    # check timesteps in decreasing order
    assert all(
        [timesteps[i] > timesteps[i + 1] for i in range(len(timesteps) - 1)]
    ), f"timesteps must be in decreasing order, got {timesteps}"

    noisy_latents_ = noisy_latents.clone()
    for t_ in timesteps:
        noise_pred = predict_noise(
            unet=unet,
            noisy_latents=noisy_latents_,
            text_embeddings=text_embeddings,
            uncond_text_embeddings=uncond_text_embeddings,
            t=t_,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
        )

        # update noisy_latents_
        noisy_latents_ = scheduler.step(
            noise_pred, t_, noisy_latents_
        ).pred_original_sample

    sample_prediction = noisy_latents_

    return sample_prediction


# based on huggingface diffusers
def ddim_step(
    noise_pred: Optional[torch.FloatTensor],
    t: int,
    xt: Optional[torch.FloatTensor] = None,
    x0: Optional[torch.FloatTensor] = None,
    *,
    eta: float = 0.0,
    num_train_timesteps=1000,
    num_inference_steps=50,
    alphas_cumprod=None,
):
    """
    Performs a single step of the DDIM scheduler.
    Args:
        noise_pred: model_output from the unet
        t: timestep
        xt: sample at timestep t
        x0: if provided, will be used as the "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        eta: weight of noise for added noise in diffusion step.
        num_train_timesteps: number of diffusion training timesteps
        num_inference_steps: number of diffusion inference timesteps
        alphas_cumprod: see scheduler.alphas_cumprod

    Returns:
        prev_sample, x0
    """
    assert xt is not None or x0 is not None, "either xt or x0 must be provided"
    assert alphas_cumprod is not None, "alphas_cumprod must be provided"

    device = noise_pred.device

    prev_t = t - num_train_timesteps // num_inference_steps

    alphas_cumprod = alphas_cumprod.to(device)

    alpha_prod_t = alphas_cumprod[t].to(device)
    alpha_prod_t_prev = (
        alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
    )

    beta_prod_t = 1 - alpha_prod_t

    if (
        x0 is None
    ):  # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        x0 = (xt - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    std_dev_t = eta * variance ** (0.5)
    # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred
    # compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * x0 + pred_sample_direction

    if eta > 0:  # add random noise
        variance_noise = torch.randn_like(noise_pred).to(device)
        variance = std_dev_t * variance_noise

        prev_sample = prev_sample + variance

    return prev_sample, x0


def ddim_inverse_step(scheduler, x_t, e_t, t, delta_t):
    r"""

    Args:
        scheduler:
        x_t: noisy latent
        e_t: noise prediction
        t:
        delta_t:

    Returns:
        x_next: x_{t+1}
        pred_x0:
    """
    next_t = t + delta_t
    a_t = scheduler.alphas_cumprod[t]
    a_next = scheduler.alphas_cumprod[next_t] if next_t <= 999 else torch.tensor(0.0)
    sqrt_one_minus_a_t = torch.sqrt(1 - a_t)
    # current prediction for x_0
    pred_x0 = (x_t - sqrt_one_minus_a_t * e_t) / a_t.sqrt()
    # direction pointing to x_t
    dir_xt = (1.0 - a_next).sqrt() * e_t
    # Equation 12. reversed
    x_next = a_next.sqrt() * pred_x0 + dir_xt
    return x_next, pred_x0
