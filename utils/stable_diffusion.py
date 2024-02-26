import torch
from diffusers import AutoencoderKL


def encode_images(images: torch.FloatTensor, vae: AutoencoderKL):
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


def decode_latents(latents: torch.FloatTensor, vae: AutoencoderKL):
    """
    Decodes latents using the VAE.

    Returns: images, range [0, 1]
    """
    latents = latents.to(vae.device)
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    return torch.clamp((images + 1.0) / 2.0, 0, 1)


def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={},
                            scheduler=None, lora_v=False, half_inference=False):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if lora_v:
        # https://github.com/threestudio-project/threestudio/blob/77de7d75c34e29a492f2dda498c65d2fd4a767ff/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L512
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=noisy_latents.device, dtype=noisy_latents.dtype
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    # Convert inputs to half precision
    if half_inference:
        noisy_latents = noisy_latents.clone().half()
        text_embeddings = text_embeddings.clone().half()
        latent_model_input = latent_model_input.clone().half()
    if guidance_scale == 1.:
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:],
                          cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
    else:
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
                          cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1,
                                                                                   1) + noise_pred * torch.cat(
                [alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.float()
    return noise_pred


def predict_noise0_diffuser_multistep(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5,
                                      cross_attention_kwargs={}, scheduler=None, steps=1, eta=0, half_inference=False):
    latents = noisy_latents
    # get sub-sequence with length step_size
    t_start = t.item()
    # Ensure that t and steps are within the valid range
    if not (0 < t_start <= 1000):
        raise ValueError(f"t must be between 0 and 1000, get {t_start}")
    if t_start > steps:
        # Calculate the step size
        step_size = t_start // steps
        # Generate a list of indices
        indices = [int((steps - i) * step_size) for i in range(steps)]
        if indices[0] != t_start:
            # indices.insert(0, t_start)    # add start point
            indices[0] = t_start  # replace start point
    else:
        indices = [int((t_start - i)) for i in range(t_start)]
    if indices[-1] != 0:
        indices.append(0)
    # run multistep ddim sampling
    for i in range(len(indices)):
        t = torch.tensor([indices[i]] * t.shape[0], device=t.device)
        noise_pred = predict_noise0_diffuser(unet, latents, text_embeddings, t, guidance_scale=guidance_scale,
                                             cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler,
                                             half_inference=half_inference)
        pred_latents = scheduler.step(noise_pred, t, latents).pred_original_sample
        if indices[i + 1] == 0:
            ### use pred_latents and latents calculate equivalent noise_pred
            alpha_bar_t_start = scheduler.alphas_cumprod[indices[0]].clone().detach()
            return (noisy_latents - torch.sqrt(alpha_bar_t_start) * pred_latents) / (torch.sqrt(1 - alpha_bar_t_start))
        alpha_bar = scheduler.alphas_cumprod[indices[i]].clone().detach()
        alpha_bar_prev = scheduler.alphas_cumprod[indices[i + 1]].clone().detach()
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(latents)
        mean_pred = (
                pred_latents * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(latents.shape) - 1)))
        )  # no noise when t == 0
        latents = mean_pred + nonzero_mask * sigma * noise
    # return out["pred_xstart"]
