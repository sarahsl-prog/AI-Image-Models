"""Download the real COCO val2017 images that correspond to the prompts in
coco_prompts.json, saving them index-matched so that 0000.jpg is the real
image for prompt index 0, 0001.jpg for index 1, and so on.

Run recover_coco_prompts.py first, then run this script:
    python scripts/coco_scripts/get_coco_samples.py
    python scripts/coco_scripts/get_coco_samples.py --output-dir data/coco_samples
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main(prompts_path: str = "data/coco_prompts.json",
         output_dir: str = "data/coco_samples"):

    prompts_file = Path(prompts_path)
    if not prompts_file.exists():
        print(f"coco_prompts.json not found at {prompts_file}.")
        print("Run scripts/coco_scripts/recover_coco_prompts.py first.")
        sys.exit(1)

    prompts = json.loads(prompts_file.read_text())
    # Build {image_id: index} lookup from the prompts
    id_to_index = {entry["image_id"]: entry["index"] for entry in prompts}
    target_ids  = set(id_to_index.keys())
    print(f"Loaded {len(target_ids)} target image_ids from {prompts_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Skip image_ids that are already downloaded
    already_done = set()
    for entry in prompts:
        out = output_path / f'{entry["index"]:04d}.jpg'
        if out.exists():
            already_done.add(entry["image_id"])

    remaining = target_ids - already_done
    if not remaining:
        print(f"{output_dir}/ already has all {len(target_ids)} images, nothing to do.")
        return

    print(f"{len(already_done)} already downloaded, fetching {len(remaining)} more …")

    ds = load_dataset("phiyodr/coco2017", split="validation", streaming=True)

    saved    = 0
    seen_ids = set()   # deduplicate within the stream (multiple captions per image)

    with tqdm(total=len(remaining), unit="img") as pbar:
        for item in ds:
            image_id = item.get("image_id")

            if image_id in seen_ids:
                continue
            seen_ids.add(image_id)

            if image_id not in remaining:
                continue

            img = item.get("image")
            if img is None:
                continue

            if img.mode != "RGB":
                img = img.convert("RGB")

            idx      = id_to_index[image_id]
            out_path = output_path / f"{idx:04d}.jpg"
            img.save(out_path)
            saved += 1
            pbar.update(1)

            if saved >= len(remaining):
                break

    total = len(already_done) + saved
    print(f"Done. {saved} downloaded, {total}/{len(target_ids)} total in {output_dir}/")

    missing = len(target_ids) - total
    if missing:
        print(f"Warning: {missing} image_ids not found in the dataset stream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download COCO val2017 images matching coco_prompts.json")
    parser.add_argument("--prompts-path", default="data/coco_prompts.json",
                        help="Path to coco_prompts.json (default: data/coco_prompts.json)")
    parser.add_argument("--output-dir", default="data/coco_samples",
                        help="Where to save images (default: data/coco_samples)")
    args = parser.parse_args()
    main(args.prompts_path, args.output_dir)
