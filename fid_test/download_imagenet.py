"""Download n random images from ImageNet (ILSVRC2012) via Hugging Face Hub.

Usage:
    uv run download_imagenet.py --n 1000 --output-dir imagenet_samples
    uv run download_imagenet.py --n 1000 --split validation
"""

import argparse
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download random ImageNet images from HF Hub")
    parser.add_argument("--n", type=int, default=1000, help="Number of images to download")
    parser.add_argument("--output-dir", type=str, default="imagenet_samples", help="Output directory")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (train/validation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ImageNet {args.split} split (streaming)...")
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split=args.split,
        streaming=True,
    )

    # Collect all indices, then sample
    # For validation split (50k images), we can just shuffle and take n
    # Streaming + shuffle with a buffer is the memory-efficient approach
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    print(f"Downloading {args.n} random images to {output_dir}/")

    count = 0
    for sample in ds:
        if count >= args.n:
            break

        image = sample["image"]
        label = sample["label"]

        path = output_dir / f"{count:05d}_label{label}.png"
        image.save(path)

        count += 1
        if count % 100 == 0:
            print(f"  [{count}/{args.n}]")

    print(f"Done! Downloaded {count} images to {output_dir}/")


if __name__ == "__main__":
    main()
