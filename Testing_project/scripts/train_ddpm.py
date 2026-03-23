'''
Train a small DDPM (UNet2DModel + DDPMScheduler) on a directory of images.
Designed to run on Lambda Labs (or locally) with no cloud-specific dependencies.
Uses Hugging Face Accelerate for automatic multi-GPU support.

Requires:
    pip install diffusers accelerate

Usage:
    # Single GPU (or CPU)
    python scripts/train_ddpm.py \
        --data-dir data/training_128/real \
        --output-dir checkpoints/ddpm_real \
        --run-name ddpm-real

    # Multi-GPU (all available GPUs)
    accelerate launch scripts/train_ddpm.py \
        --data-dir data/training_128/real \
        --output-dir checkpoints/ddpm_real \
        --run-name ddpm-real

    # Train on synthetic images
    accelerate launch scripts/train_ddpm.py \
        --data-dir data/training_128/synthetic \
        --output-dir checkpoints/ddpm_synthetic \
        --run-name ddpm-synthetic-flux-dev

After training, rsync checkpoints back to local machine:
    rsync -avz ubuntu@<lambda-ip>:~/checkpoints/ ./checkpoints/
'''

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from accelerate import Accelerator
    from diffusers import DDPMScheduler, UNet2DModel
except ImportError:
    print('diffusers and accelerate are required: pip install diffusers accelerate')
    sys.exit(1)

SUPPORTED = {'.png', '.jpg', '.jpeg'}


# ── dataset ───────────────────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    def __init__(self, image_dir, image_size):
        self.paths = sorted(
            p for p in Path(image_dir).rglob('*')
            if p.suffix.lower() in SUPPORTED
        )
        if not self.paths:
            raise FileNotFoundError(f'No images found in {image_dir}')
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),                        # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],        # → [-1, 1]
                                  [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))


# ── model ─────────────────────────────────────────────────────────────────────

def _build_unet(image_size: int) -> UNet2DModel:
    '''
    Small UNet suitable for 128×128 training on a single GPU.
    ~85M parameters — trainable in a few hours on an A10.
    '''
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            'DownBlock2D',
            'AttnDownBlock2D',
            'AttnDownBlock2D',
            'AttnDownBlock2D',
        ),
        up_block_types=(
            'AttnUpBlock2D',
            'AttnUpBlock2D',
            'AttnUpBlock2D',
            'UpBlock2D',
        ),
    )


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    is_main = accelerator.is_main_process

    if is_main:
        print(f'GPUs: {accelerator.num_processes}  '
              f'mixed_precision: {args.mixed_precision}')

    # ── data ──────────────────────────────────────────────────────────────────
    dataset = _ImageDataset(args.data_dir, args.image_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True)
    if is_main:
        print(f'Dataset: {len(dataset)} images  |  {len(loader)} batches/epoch')

    # ── model + scheduler ─────────────────────────────────────────────────────
    model     = _build_unet(args.image_size)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'UNet: {n_params:.1f}M parameters')

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_sched  = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # ── optional experiment tracking (main process only) ──────────────────────
    logger = None
    if is_main and not args.no_tracking:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from tracking.experiment_logger import WandbLogger
            logger = WandbLogger(run_name=args.run_name, config=vars(args))
        except Exception as e:
            print(f'Tracking unavailable ({e}), continuing without it.')

    # ── prepare for distributed training ──────────────────────────────────────
    model, optimizer, loader, lr_sched = accelerator.prepare(
        model, optimizer, loader, lr_sched
    )

    # ── output dir ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── train ─────────────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in tqdm(loader, desc=f'Epoch {epoch}/{args.num_epochs}',
                          leave=False, disable=not is_main):
            # Sample random timesteps and add noise
            noise     = torch.randn_like(batch)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch.shape[0],), device=accelerator.device
            ).long()
            noisy_batch = scheduler.add_noise(batch, noise, timesteps)

            # Predict noise and compute loss
            pred = model(noisy_batch, timesteps).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

        lr_sched.step()

        if is_main:
            mean_loss = float(np.mean(epoch_losses))
            print(f'Epoch {epoch:4d}/{args.num_epochs}  loss={mean_loss:.5f}  '
                  f'lr={lr_sched.get_last_lr()[0]:.2e}')

            if logger:
                logger.log({'train_loss': mean_loss, 'lr': lr_sched.get_last_lr()[0]},
                           step=epoch)

        # Save checkpoint every N epochs and at the end (main process only)
        if is_main and (epoch % args.save_every == 0 or epoch == args.num_epochs):
            ckpt_dir = output_dir / f'checkpoint-epoch{epoch:04d}'
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir / 'unet')
            scheduler.save_pretrained(ckpt_dir / 'scheduler')
            print(f'  Checkpoint saved → {ckpt_dir}')

    # Save final model at top level for easy loading
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir / 'unet')
        scheduler.save_pretrained(output_dir / 'scheduler')
        print(f'Final model saved → {output_dir}')

    if logger:
        logger.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description='Train a DDPM on an image directory')
    parser.add_argument('--data-dir',   required=True,
                        help='Directory of training images')
    parser.add_argument('--output-dir', required=True,
                        help='Where to save model checkpoints')
    parser.add_argument('--run-name',   default='ddpm-train',
                        help='W&B / MLflow run name')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Training resolution (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16; reduce if OOM)')
    parser.add_argument('--lr',         type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save a checkpoint every N epochs (default: 50)')
    parser.add_argument('--mixed-precision', default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training (default: fp16)')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable W&B and MLflow logging')
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
