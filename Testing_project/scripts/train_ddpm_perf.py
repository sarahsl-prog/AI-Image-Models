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

    # Enable torch.compile for faster iteration after first epoch
    accelerate launch scripts/train_ddpm.py \
        --data-dir data/training_128/real \
        --output-dir checkpoints/ddpm_real \
        --compile

    # Run PyTorch GPU profiler for N epochs (writes to output-dir/profiler/)
    accelerate launch scripts/train_ddpm.py \
        --data-dir data/training_128/real \
        --output-dir checkpoints/ddpm_real \
        --profile-epochs 2

After training, rsync checkpoints back to local machine:
    rsync -avz ubuntu@<lambda-ip>:~/checkpoints/ ./checkpoints/
'''

import argparse
import json
import logging
import sys
import time
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


# ── file logger ───────────────────────────────────────────────────────────────
# CHANGE: Added a persistent file logger so that all console output that would
# previously be lost after a run is also written to a structured log file.
# This is especially useful on remote Lambda instances where you may not have
# a terminal attached for the full run.

def _setup_file_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger('ddpm_train')
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if called more than once
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s'))
        logger.addHandler(fh)
    return logger


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


# ── profiler context ──────────────────────────────────────────────────────────
# CHANGE: Added a helper to build a PyTorch profiler that captures both CPU and
# CUDA activity and writes a Chrome-trace JSON + a key-averages text summary to
# output_dir/profiler/.  We only run the profiler for the first --profile-epochs
# epochs so it doesn't slow down the whole run.  Set --profile-epochs 0 (default)
# to skip profiling entirely.

def _make_profiler(output_dir: Path, epoch: int):
    '''Return a torch.profiler.profile context for the given epoch.'''
    trace_dir = output_dir / 'profiler'
    trace_dir.mkdir(parents=True, exist_ok=True)

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # Warm up 1 batch, active for up to 5, repeat=1 means one cycle.
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,   # set True for full stack traces (much larger files)
    )


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    is_main = accelerator.is_main_process

    # ── file logger (main process only) ───────────────────────────────────────
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        flog = _setup_file_logger(output_dir / 'train.log')
    else:
        flog = logging.getLogger('ddpm_train')   # no-op on non-main ranks

    def log(msg: str):
        '''Print to stdout AND write to logfile (main process only).'''
        if is_main:
            print(msg)
            flog.info(msg)

    log(f'GPUs: {accelerator.num_processes}  mixed_precision: {args.mixed_precision}')

    # ── data ──────────────────────────────────────────────────────────────────
    dataset = _ImageDataset(args.data_dir, args.image_size)

    # CHANGE: pin_memory=True (was False).
    # When pin_memory is True, the DataLoader allocates batches in page-locked
    # (pinned) CPU memory, which allows the CUDA DMA engine to transfer data to
    # the GPU asynchronously — essentially the CPU→GPU copy overlaps with
    # compute instead of blocking it.  Meaningless on CPU-only runs, free win
    # on GPU runs.
    #
    # CHANGE: num_workers default changed from 0 → 4 (see CLI section).
    # num_workers=0 means the main process loads every batch synchronously,
    # so the GPU sits idle waiting for data.  With N workers the next batch is
    # prefetched in background processes while the GPU processes the current one.
    # Rule of thumb: start at 4; go up to num_CPU_cores/num_GPUs if still bound.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,         # was: pin_memory=False
        persistent_workers=args.num_workers > 0,  # keep workers alive between epochs
    )
    log(f'Dataset: {len(dataset)} images  |  {len(loader)} batches/epoch')

    # ── model + scheduler ─────────────────────────────────────────────────────
    model     = _build_unet(args.image_size)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    log(f'UNet: {n_params:.1f}M parameters')

    # CHANGE: Optional torch.compile() via --compile flag.
    # torch.compile() (PyTorch 2.0+) runs the model through TorchDynamo and
    # Triton to generate fused CUDA kernels.  Typical gains are 10–30% on
    # throughput after the first epoch (which is slow due to compilation).
    # Gated behind a flag because: (a) it adds ~1–3 min of compile time on
    # first epoch, (b) it can conflict with some debuggers / older CUDA versions.
    # Not applied before accelerator.prepare() so Accelerate can still wrap it.
    if args.compile:
        log('torch.compile() enabled — first epoch will be slow (compilation)')
        model = torch.compile(model)

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
            log(f'Tracking unavailable ({e}), continuing without it.')

    # ── prepare for distributed training ──────────────────────────────────────
    model, optimizer, loader, lr_sched = accelerator.prepare(
        model, optimizer, loader, lr_sched
    )

    # ── train ─────────────────────────────────────────────────────────────────
    global_step   = 0
    run_start     = time.perf_counter()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_start = time.perf_counter()

        # CHANGE: Accumulate loss as a GPU tensor instead of calling .item()
        # inside the loop (was: epoch_losses = []; epoch_losses.append(loss.item())).
        #
        # loss.item() forces a CPU-GPU synchronisation on every single step —
        # the CPU blocks until the GPU finishes computing that scalar.  For a
        # loop with hundreds of steps this adds up fast.  Instead we keep the
        # running sum on-device and do exactly one .item() call at the end of
        # the epoch, paying the sync cost only once.
        epoch_loss_sum = torch.zeros(1, device=accelerator.device)
        n_steps        = 0

        # Timing buckets (CPU wall-clock, good enough for identifying
        # data-loading vs. compute imbalance without needing CUDA events).
        data_time  = 0.0
        step_time  = 0.0

        # ── optional PyTorch profiler ─────────────────────────────────────────
        # Profile only the first --profile-epochs epochs on the main process.
        # The profiler adds overhead so we don't run it for the whole training.
        use_profiler = is_main and (epoch <= args.profile_epochs)
        prof_ctx     = _make_profiler(output_dir, epoch) if use_profiler else None

        if use_profiler:
            log(f'[Profiler] active for epoch {epoch} — trace → {output_dir}/profiler/')
            prof_ctx.__enter__()

        t_data_start = time.perf_counter()

        for batch in tqdm(loader, desc=f'Epoch {epoch}/{args.num_epochs}',
                          leave=False, disable=not is_main):

            data_time   += time.perf_counter() - t_data_start
            t_step_start = time.perf_counter()

            # ── forward + backward ────────────────────────────────────────────
            noise     = torch.randn_like(batch)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch.shape[0],), device=accelerator.device
            ).long()
            noisy_batch = scheduler.add_noise(batch, noise, timesteps)

            pred = model(noisy_batch, timesteps).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate loss on GPU (no sync)
            epoch_loss_sum += loss.detach()
            n_steps        += 1
            global_step    += 1

            step_time += time.perf_counter() - t_step_start

            # Step the profiler scheduler (no-op when profiler is off)
            if use_profiler:
                prof_ctx.step()

            t_data_start = time.perf_counter()

        # ── end-of-epoch ──────────────────────────────────────────────────────
        if use_profiler:
            prof_ctx.__exit__(None, None, None)

            # Write a human-readable summary to the logfile in addition to the
            # binary trace that TensorBoard / Perfetto can open.
            summary = prof_ctx.key_averages().table(
                sort_by='cuda_time_total', row_limit=20
            )
            summary_path = output_dir / 'profiler' / f'epoch{epoch:04d}_summary.txt'
            summary_path.write_text(summary)
            flog.info(f'[Profiler] epoch {epoch} summary → {summary_path}')

        lr_sched.step()

        if is_main:
            # Single .item() call here — one GPU sync per epoch instead of one per step
            mean_loss  = (epoch_loss_sum / n_steps).item()
            epoch_time = time.perf_counter() - epoch_start
            total_time = time.perf_counter() - run_start

            # data_load_pct: if consistently >20% your DataLoader is the bottleneck
            # (increase num_workers).  Near 0% means the GPU is the bottleneck.
            total_iter_time = data_time + step_time
            data_pct        = (data_time / total_iter_time * 100) if total_iter_time > 0 else 0.0
            imgs_per_sec    = len(dataset) / epoch_time

            log(
                f'Epoch {epoch:4d}/{args.num_epochs}  '
                f'loss={mean_loss:.5f}  '
                f'lr={lr_sched.get_last_lr()[0]:.2e}  '
                f'epoch={epoch_time:.1f}s  '
                f'data={data_pct:.1f}%  '
                f'img/s={imgs_per_sec:.1f}'
            )

            metrics = {
                'train_loss':       mean_loss,
                'lr':               lr_sched.get_last_lr()[0],
                # ── timing metrics ──────────────────────────────────────────
                # CHANGE: These were not tracked before.  Logging them to wandb
                # lets you see trends across epochs and compare runs so you know
                # whether optimisations are actually helping.
                'perf/epoch_time_s':      epoch_time,
                'perf/data_load_time_s':  data_time,
                'perf/compute_time_s':    step_time,
                'perf/data_load_pct':     data_pct,
                'perf/imgs_per_sec':      imgs_per_sec,
                'perf/total_elapsed_s':   total_time,
                'perf/global_step':       global_step,
            }

            # Also append a JSONL line to the logfile for easy offline analysis
            # (grep / pandas / jq — your choice).
            with open(output_dir / 'metrics.jsonl', 'a') as fj:
                fj.write(json.dumps({'epoch': epoch, **metrics}) + '\n')

            if logger:
                logger.log(metrics, step=epoch)

        # ── checkpoint ────────────────────────────────────────────────────────
        if is_main and (epoch % args.save_every == 0 or epoch == args.num_epochs):
            ckpt_dir  = output_dir / f'checkpoint-epoch{epoch:04d}'
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir / 'unet')
            scheduler.save_pretrained(ckpt_dir / 'scheduler')
            log(f'  Checkpoint saved → {ckpt_dir}')

    # ── final model ───────────────────────────────────────────────────────────
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir / 'unet')
        scheduler.save_pretrained(output_dir / 'scheduler')
        total_time = time.perf_counter() - run_start
        log(f'Final model saved → {output_dir}')
        log(f'Total training time: {total_time / 60:.1f} min')

        if logger:
            logger.log({'perf/total_training_time_min': total_time / 60}, step=args.num_epochs)

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

    # CHANGE: Default changed from 0 → 4.
    # See DataLoader comment in train() for the full explanation.  Set to 0 to
    # restore the original single-process behaviour (e.g. for debugging).
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4; set 0 to disable)')

    parser.add_argument('--mixed-precision', default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training (default: fp16)')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable W&B and MLflow logging')

    # CHANGE: New flag — torch.compile() opt-in.
    # See model section in train() for details.  Not on by default because the
    # first-epoch compile cost may be surprising if you're doing quick tests.
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile() for faster training (PyTorch 2.0+; '
                             'first epoch will be slow due to compilation)')

    # CHANGE: New flag — PyTorch profiler.
    # Profiles the first N epochs and writes Chrome-trace + text summaries to
    # output-dir/profiler/.  Default 0 = disabled.  Use 1-2 epochs; more is
    # usually unnecessary and adds overhead.
    parser.add_argument('--profile-epochs', type=int, default=0,
                        help='Profile the first N epochs with torch.profiler (default: 0 = off). '
                             'Traces written to output-dir/profiler/')

    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
