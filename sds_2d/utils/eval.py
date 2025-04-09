import warnings
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipSimilarity:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initializes the CLIP model and processor from Hugging Face.

        Args:
            model_name (str): The name of the CLIP model to use from Hugging Face.
                              Defaults to "openai/clip-vit-large-patch14".
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def to_device(self, device: torch.device):
        self.device = device
        self.model = self.model.to(self.device)
        return self

    def encode_text(self, texts):
        """
        Encodes text prompts into CLIP text feature embeddings.

        Args:
            texts (list of str): A list of text prompts to encode.

        Returns:
            torch.Tensor: Normalized text embeddings.
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            self.device
        )
        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def encode_image(self, images):
        """
        Encodes images into CLIP image feature embeddings.

        Args:
            images (list of PIL.Image.Image): A list of PIL images to encode.

        Returns:
            torch.Tensor: Normalized image embeddings.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def compute_text_img_similarity(self, img, prompt):
        """
        Compute text-image similarity
        Args:
            img: (PIL.Image.Image)
            prompt:

        Returns:

        """
        image_features = self.encode_image([img])
        text_features = self.encode_text([prompt])
        return F.cosine_similarity(image_features, text_features).item()

    def compute_similarity(self, src_img, tgt_img, src_prompt, tgt_prompt):
        """
        Computes CLIP similarities between images and text prompts.

        Args:
            src_img (PIL.Image.Image): Source image.
            tgt_img (PIL.Image.Image): Target image.
            src_prompt (str): Source text prompt.
            tgt_prompt (str): Target text prompt.

        Returns:
            dict: Dictionary containing the following similarity scores:
                  - sim_0: Similarity between source image and source text.
                  - sim_1: Similarity between target image and target text.
                  - sim_direction: Directional similarity between (image_1 - image_0) and (text_1 - text_0).
                  - sim_image: Similarity between source and target images.
        """
        # Encode images and text
        src_image_features = self.encode_image([src_img])
        tgt_image_features = self.encode_image([tgt_img])
        src_text_features = self.encode_text([src_prompt])
        tgt_text_features = self.encode_text([tgt_prompt])

        # Compute similarities
        sim_0 = F.cosine_similarity(src_image_features, src_text_features).item()
        sim_1 = F.cosine_similarity(tgt_image_features, tgt_text_features).item()
        sim_direction = F.cosine_similarity(
            tgt_image_features - src_image_features,
            tgt_text_features - src_text_features,
        ).item()
        sim_image = F.cosine_similarity(src_image_features, tgt_image_features).item()

        return {
            "sim_0": sim_0,
            "sim_1": sim_1,
            "sim_direction": sim_direction,
            "sim_image": sim_image,
        }


def preprocess_image(image_path):
    """
    Loads and preprocesses an image from the given file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Preprocessed image.
    """
    image = Image.open(image_path).convert("RGB")
    return image


_clip_similarity_instances = defaultdict(
    lambda: None
)  # key: model_name, value: ClipSimilarity instance


def get_clip_similarity(model_name: str):
    """
    Get or lazily initialize a ClipSimilarity instance for a given model name.

    Args:
        model_name (str): The name of the CLIP model to use.

    Returns:
        ClipSimilarity: The ClipSimilarity instance.
    """
    global _clip_similarity_instances

    # If the instance doesn't exist, create it
    if _clip_similarity_instances[model_name] is None:
        print(f"Initializing ClipSimilarity for model: {model_name}")
        _clip_similarity_instances[model_name] = ClipSimilarity(model_name)  # noqa
    return _clip_similarity_instances[model_name]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src-img", type=str, required=True, help="Path to the source image."
    )
    parser.add_argument(
        "--tgt-img", type=str, required=True, help="Path to the target image."
    )
    parser.add_argument(
        "--src-prompt", type=str, required=True, help="Source text prompt."
    )
    parser.add_argument(
        "--tgt-prompt", type=str, required=True, help="Target text prompt."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model to use.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    args = parser.parse_args()

    if not args.verbose:
        warnings.filterwarnings("ignore")

    print(f"Loading CLIP model: {args.model_name}...")
    clip_similarity = get_clip_similarity(model_name=args.model_name)
    print(f"Loaded CLIP model: {args.model_name}")

    # Preprocess images
    print(f"Preprocessing images...")
    src_img = preprocess_image(args.src_img)
    tgt_img = preprocess_image(args.tgt_img)

    # Calculate similarities
    print(f"Calculating similarities...")
    results = clip_similarity.compute_similarity(
        src_img, tgt_img, args.src_prompt, args.tgt_prompt
    )

    # Output results
    print(f"Similarity between source image and source text: {results['sim_0']:.4f}")
    print(f"Similarity between target image and target text: {results['sim_1']:.4f}")
    print(f"Directional similarity: {results['sim_direction']:.4f}")
    print(f"Similarity between source and target images: {results['sim_image']:.4f}")
