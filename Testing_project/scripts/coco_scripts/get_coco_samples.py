"""Download the real COCO val2017 images that correspond to the prompts in
coco_prompts.json, saving them index-matched so that 0000.jpg is the real
image for prompt index 0, 0001.jpg for index 1, and so on.

Downloads directly from COCO servers — no HuggingFace dataset required.

Run recover_coco_prompts.py first, then run this script:
    python scripts/coco_scripts/get_coco_samples.py
    python scripts/coco_scripts/get_coco_samples.py --output-dir data/coco_samples
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

COCO_URL = "https://images.cocodataset.org/val2017/{image_id:012d}.jpg"


def download_image(url: str, dest: Path, session: requests.Session) -> bool:
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"\nFailed to download {url}: {e}")
        return False


def main(prompts_path: str = "data/coco_prompts.json",
         output_dir: str = "data/coco_samples"):

    prompts_file = Path(prompts_path)
    if not prompts_file.exists():
        print(f"coco_prompts.json not found at {prompts_file}.")
        print("Run scripts/coco_scripts/recover_coco_prompts.py first.")
        sys.exit(1)

    prompts = json.loads(prompts_file.read_text())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Identify which indices still need downloading
    todo = [
        entry for entry in prompts
        if not (output_path / f'{entry["index"]:04d}.jpg').exists()
    ]

    if not todo:
        print(f"{output_dir}/ already has all {len(prompts)} images, nothing to do.")
        return

    already = len(prompts) - len(todo)
    print(f"{already} already downloaded, fetching {len(todo)} more …")

    failed = 0
    with requests.Session() as session:
        for entry in tqdm(todo, unit="img"):
            url  = COCO_URL.format(image_id=entry["image_id"])
            dest = output_path / f'{entry["index"]:04d}.jpg'
            if not download_image(url, dest, session):
                failed += 1

    saved = len(todo) - failed
    total = already + saved
    print(f"Done. {saved} downloaded, {total}/{len(prompts)} total in {output_dir}/")
    if failed:
        print(f"Warning: {failed} images failed to download.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download COCO val2017 images matching coco_prompts.json")
    parser.add_argument("--prompts-path", default="data/coco_prompts.json",
                        help="Path to coco_prompts.json (default: data/coco_prompts.json)")
    parser.add_argument("--output-dir", default="data/coco_samples",
                        help="Where to save images (default: data/coco_samples)")
    args = parser.parse_args()
    main(args.prompts_path, args.output_dir)
