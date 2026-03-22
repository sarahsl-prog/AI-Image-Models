'''
Generate sample images from a trained DDPM checkpoint.
Saves output as 0000.png, 0001.png, ... for compatibility with evaluate_fid().

Requires:
    pip install diffusers accelerate

Usage:
    python scripts/generate_ddpm_samples.py \
        --checkpoint-dir checkpoints/ddpm_real \
        --output-dir data/ddpm_samples/real \
        --num-samples 1000

    python scripts/generate_ddpm_samples.py \
        --checkpoint-dir checkpoints/ddpm_synthetic \
        --output-dir data/ddpm_samples/synthetic \
        --num-samples 1000
'''

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
except ImportError:
    print('diffusers is required: pip install diffusers accelerate')
    sys.exit(1)


def generate_samples(checkpoint_dir: Path, output_dir: Path,
                     num_samples: int, batch_size: int):
    '''
    Load a UNet + DDPMScheduler from checkpoint_dir and generate num_samples
    images, saving them as zero-padded PNGs in output_dir.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Loading checkpoint from {checkpoint_dir} …')

    unet      = UNet2DModel.from_pretrained(checkpoint_dir / 'unet').to(device)
    scheduler = DDPMScheduler.from_pretrained(checkpoint_dir / 'scheduler')
    pipeline  = DDPMPipeline(unet=unet, scheduler=scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    pbar = tqdm(total=num_samples, desc='Generating', unit='img')

    while generated < num_samples:
        batch = min(batch_size, num_samples - generated)
        images = pipeline(batch_size=batch, output_type='pil').images

        for img in images:
            out_path = output_dir / f'{generated:04d}.png'
            img.save(out_path)
            generated += 1
            pbar.update(1)

    pbar.close()
    print(f'{generated} images saved → {output_dir}')


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate images from a trained DDPM checkpoint')
    parser.add_argument('--checkpoint-dir', required=True,
                        help='Path to trained checkpoint (contains unet/ and scheduler/)')
    parser.add_argument('--output-dir', required=True,
                        help='Where to save generated images')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of images to generate (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Generation batch size (default: 16)')
    return parser.parse_args()


def main():
    args = _parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    if not (checkpoint_dir / 'unet').exists():
        print(f'No unet/ found in {checkpoint_dir}. '
              f'Run train_ddpm.py first, or check the checkpoint path.')
        sys.exit(1)

    generate_samples(
        checkpoint_dir=checkpoint_dir,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    print('Done.')


if __name__ == '__main__':
    main()
