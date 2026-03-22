'''
Resize images to a target resolution before uploading to a remote training
instance (e.g. Lambda Labs). Preserves filenames and directory structure.

Usage:
    # Resize one directory
    python scripts/resize_for_training.py \
        --src data/generated_images/black-forest-labs--FLUX.1-dev/coco \
        --dst data/training_128/synthetic \
        --size 128

    # Resize both real and synthetic in one call (loop 4 helper)
    python scripts/resize_for_training.py \
        --src data/coco_samples \
        --dst data/training_128/real \
        --size 128

    python scripts/resize_for_training.py \
        --src data/generated_images/black-forest-labs--FLUX.1-dev/coco \
        --dst data/training_128/synthetic \
        --size 128
'''

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SUPPORTED = {'.png', '.jpg', '.jpeg'}


def resize_directory(src: Path, dst: Path, size: int, quality: int = 95):
    '''
    Resize all images in src to size×size and write to dst.
    Uses LANCZOS resampling and saves as PNG to avoid compression artefacts
    affecting FID computation.
    '''
    dst.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src.rglob('*') if p.suffix.lower() in SUPPORTED)
    if not image_paths:
        print(f'No images found in {src}')
        return 0

    skipped = 0
    for src_path in tqdm(image_paths, desc=f'{src.name} → {dst.name}', unit='img'):
        # Preserve relative subdirectory structure
        rel      = src_path.relative_to(src)
        dst_path = (dst / rel).with_suffix('.png')
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            skipped += 1
            continue

        img = Image.open(src_path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst_path)

    done = len(image_paths) - skipped
    print(f'  {done} resized, {skipped} already existed → {dst}')
    return done


def _parse_args():
    parser = argparse.ArgumentParser(description='Resize images for DDPM training')
    parser.add_argument('--src',  required=True, help='Source image directory')
    parser.add_argument('--dst',  required=True, help='Destination directory')
    parser.add_argument('--size', type=int, default=128,
                        help='Target width and height in pixels (default: 128)')
    return parser.parse_args()


def main():
    args = _parse_args()
    src  = Path(args.src)
    dst  = Path(args.dst)

    if not src.exists():
        print(f'Source directory not found: {src}')
        sys.exit(1)

    resize_directory(src, dst, args.size)
    print('Done.')


if __name__ == '__main__':
    main()
