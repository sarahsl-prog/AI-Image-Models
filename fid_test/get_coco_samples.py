"""Download real COCO 2017 validation images for use as the reference set in eval_std.py.

Run once before evaluating COCO-track results:
    python get_coco_samples.py
    python get_coco_samples.py --num-images 1000 --output-dir coco_samples
"""

import argparse
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image
from datasets import load_dataset


def main(num_images: int = 2500, output_dir: str = "coco_samples"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_path.glob("*.jpg")))
    if existing >= num_images:
        print(f"{output_dir}/ already has {existing} images, nothing to do.")
        return

    print(f"Downloading {num_images} COCO val2017 images to {output_dir}/...")
    ds = load_dataset("phiyodr/coco2017", split="validation", streaming=True)

    count = 0
    seen_ids = set()
    for item in ds:
        image_id = item["image_id"]
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        try:
            with urllib.request.urlopen(item["coco_url"]) as resp:
                img = Image.open(BytesIO(resp.read()))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(output_path / f"{count:04d}.jpg")
        except Exception as e:
            print(f"  Warning: skipping {item['coco_url']}: {e}")
            continue

        count += 1
        if count % 100 == 0:
            print(f"  {count}/{num_images}")
        if count >= num_images:
            break

    print(f"Done! Saved {count} images to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--output-dir", default="coco_samples")
    args = parser.parse_args()
    main(args.num_images, args.output_dir)
