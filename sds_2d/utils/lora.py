from contextlib import contextmanager

from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)


@contextmanager
def default_attn_processor(unet: UNet2DConditionModel):
    """
    Temporarily set the default attention processor for the given unet.
    Args:
        unet: UNet2DConditionModel
    """
    attn_processors = unet.attn_processors
    unet.set_default_attn_processor()
    yield
    unet.set_attn_processor(attn_processors)


def set_lora_(unet):
    """
    Add LoRA to the given unet in-place.
    Args:
        unet:

    Returns:

    """
    ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
    ### begin lora
    # Set correct lora layers
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        hidden_size = None
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor

        if hidden_size is not None:
            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).to(unet.device)

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    for param in unet_lora_layers.parameters():
        param.requires_grad_(True)

    ### end lora
    return unet, unet_lora_layers
