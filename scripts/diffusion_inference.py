from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
from tqdm import tqdm

##################################################
# configs
device = "cuda"
samples_per_prompt = 4

guidance = 7.5

prompts = [
    "a photograph of an astronaut riding a horse.",
    "a cat on a wooden table.",
]
##################################################

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", local_files_only=True, cache_dir="./.cache"
)

pipe = pipe.to(device)

fig, axes = plt.subplots(len(prompts), samples_per_prompt, figsize=(20, 20))

for i, prompt in tqdm(enumerate(prompts), desc="prompts"):
    prompt_stripped = prompt.strip()
    print(f"prompt: {prompt}")
    for j in tqdm(range(samples_per_prompt), desc="samples"):
        image = pipe(
            prompt,
            guidance_scale=guidance,
        ).images[0]
        axes[i, j].imshow(image)

plt.show()
