"""Generate images from COCO captions using HuggingFace diffusion models on Modal.

Usage:
    modal run make_images_coco.py --model stabilityai/stable-diffusion-xl-base-1.0 --num-images 1000
    modal run make_images_coco.py --model runwayml/stable-diffusion-v1-5 --num-images 1000
    modal run make_images_coco.py --model black-forest-labs/FLUX.1-dev --num-images 1000
    modal run make_images_coco.py --model black-forest-labs/FLUX.1-schnell --num-images 1000
    modal run make_images_coco.py --model stabilityai/stable-diffusion-3.5-large --num-images 1000
    modal run make_images_coco.py --model stabilityai/sdxl-turbo --num-images 1000

Saves to: generated_images/{model-slug}/coco/
Change GPU below in @app.cls if you need more VRAM (e.g. "A100" for Flux models).
"""

import io
import random
from pathlib import Path

import modal

MINUTES = 60
CACHE_DIR = "/cache"

app = modal.App("make-coco-images")

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


def get_coco_captions(n: int, seed: int = 42) -> list[str]:
    """Fetch captions from the COCO 2017 validation set via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("phiyodr/coco2017", split="validation", streaming=True)
    captions = []
    for item in ds:
        # phiyodr/coco2017 stores captions as a dict: {"raw": [...], "processed": [...]}
        raw = item.get("captions", {}).get("raw", [])
        if raw:
            captions.append(raw[0])
        if len(captions) >= n * 2:  # gather extra so we can shuffle and sample
            break

    random.seed(seed)
    random.shuffle(captions)
    return captions[:n]


@app.local_entrypoint()
def main(
    model: str,
    num_images: int = 1000,
):
    print(f"Fetching {num_images} COCO captions...")
    captions = get_coco_captions(num_images)
    print(f"Got {len(captions)} captions")

    model_slug = model.replace("/", "--")
    output_dir = Path(f"generated_images/{model_slug}/coco")

    print(f"Generating {num_images} images with {model}")
    print(f"Saving to {output_dir}")

    generator = Generator()

    for i, image_bytes in enumerate(
        generator.generate.map(
            [model] * num_images,
            captions,
            order_outputs=True,
        )
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(output_dir.glob("*.png")))
        path = output_dir / f"{existing:04d}.png"
        path.write_bytes(image_bytes)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{num_images}] saved {path}")

    print(f"Done! Generated {num_images} images in {output_dir}")
