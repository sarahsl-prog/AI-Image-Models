'''
Generate sample images from a trained DDPM checkpoint.
Saves output as 0000.png, 0001.png, ... for compatibility with evaluate_fid().

Requires:
    pip install diffusers accelerate

── Local usage ───────────────────────────────────────────────────────────────
    python scripts/generate_ddpm_samples.py \
        --checkpoint-dir checkpoints/ddpm_real \
        --output-dir data/ddpm_samples/real \
        --num-samples 2500

── Modal usage ───────────────────────────────────────────────────────────────
Upload your checkpoint to a Modal Volume first, then run:

    # One-time: create the volume
    modal volume create ddpm-data

    # Upload checkpoint(s)
    modal volume put ddpm-data checkpoints/ddpm_real       /checkpoints/ddpm_real
    modal volume put ddpm-data checkpoints/ddpm_synthetic_flux-dev /checkpoints/ddpm_synthetic_flux-dev

    # Run generation on Modal GPU (10 parallel containers by default)
    modal run scripts/generate_ddpm_samples.py \
        -- --checkpoint-dir /checkpoints/ddpm_real \
           --output-dir /samples/real \
           --num-samples 2500

    # Download results
    modal volume get ddpm-data /samples/real data/ddpm_samples/real

── Scheduler notes ───────────────────────────────────────────────────────────
    --scheduler ddim  (default) uses DDIMScheduler, which produces comparable
    quality in ~50 steps vs 200 for DDPM. Recommended for speed.

    --scheduler ddpm  uses the original DDPMScheduler from the checkpoint.
    Pair with a higher --num-inference-steps (100-200) for best quality.
'''

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from diffusers import (DDIMPipeline, DDIMScheduler, DDPMPipeline,
                           DDPMScheduler, UNet2DModel)
except ImportError:
    print('diffusers is required: pip install diffusers accelerate')
    sys.exit(1)

# ── Modal setup (optional) ────────────────────────────────────────────────────

try:
    import modal
    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False

if _MODAL_AVAILABLE:
    app = modal.App('generate-ddpm-samples')

    _volume = modal.Volume.from_name('ddpm-data', create_if_missing=True)

    _image = (
        modal.Image.debian_slim(python_version='3.11')
        .pip_install('torch', 'torchvision', 'diffusers', 'accelerate', 'tqdm')
    )


# ── core generation logic ─────────────────────────────────────────────────────

def generate_chunk(checkpoint_dir: Path, output_dir: Path,
                   start_idx: int, end_idx: int,
                   batch_size: int, num_inference_steps: int,
                   scheduler_type: str):
    '''
    Load a UNet from checkpoint_dir and generate images numbered
    [start_idx, end_idx), saving them as zero-padded PNGs in output_dir.

    Uses DDIMScheduler by default (fast, ~50 steps). Pass scheduler_type='ddpm'
    to use the original DDPMScheduler from the checkpoint.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  |  range: [{start_idx}, {end_idx})  '
          f'|  scheduler: {scheduler_type}  |  steps: {num_inference_steps}')

    unet = UNet2DModel.from_pretrained(checkpoint_dir / 'unet').to(device)

    if scheduler_type == 'ddim':
        base = DDPMScheduler.from_pretrained(checkpoint_dir / 'scheduler')
        scheduler = DDIMScheduler(
            num_train_timesteps=base.config.num_train_timesteps,
            beta_start=base.config.beta_start,
            beta_end=base.config.beta_end,
            beta_schedule=base.config.beta_schedule,
        )
        pipeline = DDIMPipeline(unet=unet, scheduler=scheduler).to(device)
    else:
        scheduler = DDPMScheduler.from_pretrained(checkpoint_dir / 'scheduler')
        pipeline = DDPMPipeline(unet=unet, scheduler=scheduler).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only generate indices that don't already exist (safe to resume/retry)
    todo = [i for i in range(start_idx, end_idx)
            if not (output_dir / f'{i:04d}.png').exists()]

    if not todo:
        print(f'All images in [{start_idx}, {end_idx}) already exist, skipping.')
        return

    print(f'Generating {len(todo)} images ({num_inference_steps} steps each) …')
    pbar = tqdm(total=len(todo), desc=f'[{start_idx}-{end_idx})', unit='img')

    pos = 0
    while pos < len(todo):
        batch_indices = todo[pos:pos + batch_size]
        generator = torch.Generator(device=device).manual_seed(batch_indices[0])
        images = pipeline(
            batch_size=len(batch_indices),
            num_inference_steps=num_inference_steps,
            output_type='pil',
            generator=generator,
        ).images

        for idx, img in zip(batch_indices, images):
            img.save(output_dir / f'{idx:04d}.png')
            pbar.update(1)

        pos += len(batch_indices)

    pbar.close()
    print(f'Chunk [{start_idx}, {end_idx}) done → {output_dir}')


# ── Modal remote function ─────────────────────────────────────────────────────

if _MODAL_AVAILABLE:
    @app.function(
        gpu='A10G',
        image=_image,
        volumes={'/data': _volume},
        timeout=3600,
    )
    def generate_on_modal(checkpoint_dir: str, output_dir: str,
                          start_idx: int, end_idx: int,
                          batch_size: int, num_inference_steps: int,
                          scheduler_type: str):
        ckpt = Path('/data') / checkpoint_dir.lstrip('/')
        out  = Path('/data') / output_dir.lstrip('/')
        generate_chunk(
            checkpoint_dir=ckpt,
            output_dir=out,
            start_idx=start_idx,
            end_idx=end_idx,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            scheduler_type=scheduler_type,
        )
        _volume.commit()

    @app.local_entrypoint()
    def modal_main(
        checkpoint_dir: str,
        output_dir: str,
        num_samples: int = 2500,
        batch_size: int = 32,
        num_inference_steps: int = 50,
        scheduler: str = 'ddim',
        num_containers: int = 10,
    ):
        chunk_size = (num_samples + num_containers - 1) // num_containers
        chunks = [
            (i, min(i + chunk_size, num_samples))
            for i in range(0, num_samples, chunk_size)
        ]
        actual_containers = len(chunks)

        print(f'Dispatching to Modal ({actual_containers} × A10G) …')
        print(f'  checkpoint       : {checkpoint_dir}')
        print(f'  output           : {output_dir}')
        print(f'  samples          : {num_samples}')
        print(f'  scheduler        : {scheduler}')
        print(f'  inference steps  : {num_inference_steps}')
        print(f'  containers       : {actual_containers}  '
              f'(~{chunk_size} images each)')

        list(generate_on_modal.starmap(
            [
                (checkpoint_dir, output_dir, start, end,
                 batch_size, num_inference_steps, scheduler)
                for start, end in chunks
            ]
        ))

        print(f'\nDone. Download results with:')
        print(f'  modal volume get ddpm-data {output_dir} '
              f'data/ddpm_samples/{Path(output_dir).name}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate images from a trained DDPM checkpoint')
    parser.add_argument('--checkpoint-dir', required=True,
                        help='Path to trained checkpoint (contains unet/ and scheduler/)')
    parser.add_argument('--output-dir', required=True,
                        help='Where to save generated images')
    parser.add_argument('--num-samples', type=int, default=2500,
                        help='Number of images to generate (default: 2500)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Generation batch size (default: 32)')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='Denoising steps per image — '
                             '50 (ddim) or 200 (ddpm) are typical (default: 50)')
    parser.add_argument('--scheduler', default='ddim', choices=['ddim', 'ddpm'],
                        help='Scheduler type: ddim is ~4x faster at equal quality '
                             '(default: ddim)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='First image index to generate (default: 0). '
                             'Useful for manual parallelism across machines.')
    return parser.parse_args()


def main():
    args = _parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    if not (checkpoint_dir / 'unet').exists():
        print(f'No unet/ found in {checkpoint_dir}. '
              f'Run train_ddpm.py first, or check the checkpoint path.')
        sys.exit(1)

    generate_chunk(
        checkpoint_dir=checkpoint_dir,
        output_dir=Path(args.output_dir),
        start_idx=args.start_idx,
        end_idx=args.start_idx + args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        scheduler_type=args.scheduler,
    )
    print('Done.')


if __name__ == '__main__':
    main()
