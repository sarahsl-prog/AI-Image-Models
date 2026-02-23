"""Generate images of ImageNet classes using HuggingFace diffusion models on Modal.

Usage:
    modal run make_images_imagenet.py --model stabilityai/stable-diffusion-xl-base-1.0 --num-images 1000
    modal run make_images_imagenet.py --model runwayml/stable-diffusion-v1-5 --num-images 1000
    modal run make_images_imagenet.py --model black-forest-labs/FLUX.1-dev --num-images 1000
    modal run make_images_imagenet.py --model black-forest-labs/FLUX.1-schnell --num-images 1000
    modal run make_images_imagenet.py --model stabilityai/stable-diffusion-3.5-large --num-images 1000
    modal run make_images_imagenet.py --model stabilityai/sdxl-turbo --num-images 1000

Saves to: generated_images/{model-slug}/imagenet/
Change GPU below in @app.cls if you need more VRAM (e.g. "A100" for Flux models).
"""

import io
import json
import urllib.request
from pathlib import Path

import modal

MINUTES = 60
CACHE_DIR = "/cache"

app = modal.App("make-imagenet-images")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "huggingface-hub==0.36.0",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
        "safetensors",
        "protobuf",
    )
    .env(
        {
            "HF_HUB_CACHE": CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

with image.imports():
    import torch
    from diffusers import DiffusionPipeline

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)


@app.cls(
    image=image,
    gpu="L40S",
    timeout=10 * MINUTES,
    container_idle_timeout=5 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
    secrets=[huggingface_secret],
)
class Generator:
    @modal.method()
    def generate(self, model_id: str, prompt: str) -> bytes:
        # lazy load: persists across calls on the same container
        if not hasattr(self, "_pipe") or self._model_id != model_id:
            self._pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            ).to("cuda")
            self._model_id = model_id

        result = self._pipe(prompt, num_images_per_prompt=1).images[0]

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()


IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def get_imagenet_classes() -> list[str]:
    """Fetch the 1000 ImageNet-1K class labels."""
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as resp:
        return json.loads(resp.read().decode())


@app.local_entrypoint()
def main(
    model: str,
    num_images: int = 1000,
):
    classes = get_imagenet_classes()

    # build prompts, cycling through classes
    prompts = []
    class_labels = []
    for i in range(num_images):
        cls = classes[i % len(classes)]
        prompts.append(f"a photo of a {cls}")
        class_labels.append(cls)

    model_slug = model.replace("/", "--")
    output_dir = Path(f"generated_images/{model_slug}/imagenet")

    print(f"Generating {num_images} images with {model}")
    print(f"Saving to {output_dir}")

    generator = Generator()

    for i, image_bytes in enumerate(
        generator.generate.map(
            [model] * num_images,
            prompts,
            order_outputs=True,
        )
    ):
        cls = class_labels[i]
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(cls_dir.glob("*.png")))
        path = cls_dir / f"{existing:04d}.png"
        path.write_bytes(image_bytes)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{num_images}] saved {path}")

    print(f"Done! Generated {num_images} images in {output_dir}")
