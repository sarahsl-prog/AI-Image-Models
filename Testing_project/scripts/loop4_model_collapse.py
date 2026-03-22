'''
Loop 4 — Model Collapse Experiment

Tests the model collapse hypothesis: does training a generative model on
synthetic images rather than real ones produce a measurably degraded
output distribution?

Experiment:
  1. Train DDPM-real     on real COCO images        → checkpoints/ddpm_real/
  2. Train DDPM-synth    on synthetic images         → checkpoints/ddpm_<model>/
  3. Generate samples from both DDPMs
  4. Compute FID(ddpm_real_samples    vs real images)  ← baseline
     Compute FID(ddpm_synth_samples   vs real images)  ← collapse indicator
  5. Save comparison chart + results JSON
     Log everything to W&B + MLflow

Lambda Labs workflow:
  # --- LOCAL: prepare and upload ---
  python scripts/loop4_model_collapse.py --synthetic-model flux-dev --prepare-only

  rsync -avz data/training_128/ ubuntu@<ip>:~/data/training_128/
  rsync -avz scripts/train_ddpm.py ubuntu@<ip>:~/scripts/

  # --- ON LAMBDA: train both models ---
  python scripts/train_ddpm.py --data-dir data/training_128/real     --output-dir checkpoints/ddpm_real     --run-name ddpm-real
  python scripts/train_ddpm.py --data-dir data/training_128/synthetic --output-dir checkpoints/ddpm_synthetic --run-name ddpm-synthetic-flux-dev

  rsync -avz ubuntu@<ip>:~/checkpoints/ ./checkpoints/

  # --- LOCAL: generate samples, evaluate, and report ---
  python scripts/loop4_model_collapse.py --synthetic-model flux-dev --skip-training

Usage:
  # Full local run (slow without a strong GPU)
  python scripts/loop4_model_collapse.py --synthetic-model flux-dev

  # Lambda workflow - data preparation only
  python scripts/loop4_model_collapse.py --synthetic-model flux-dev --prepare-only

  # After downloading checkpoints from Lambda
  python scripts/loop4_model_collapse.py --synthetic-model flux-dev --skip-training

  # Skip both training and generation (checkpoints + samples already exist)
  python scripts/loop4_model_collapse.py --synthetic-model flux-dev --skip-training --skip-generation
'''

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.fid import evaluate_fid

PROJECT_ROOT   = Path(__file__).parent.parent
DATA_DIR       = PROJECT_ROOT / 'data'
CHECKPOINTS    = PROJECT_ROOT / 'checkpoints'
REPORT_DIR     = PROJECT_ROOT / 'report' / 'model_collapse'
RESULTS_DIR    = PROJECT_ROOT / 'report' / 'results'

MODELS = {
    'flux-dev':     'black-forest-labs--FLUX.1-dev',
    'flux-schnell': 'black-forest-labs--FLUX.1-schnell',
    'sd15':         'runwayml--stable-diffusion-v1-5',
    'sdxl':         'stabilityai--stable-diffusion-xl-base-1.0',
}


# ── step 1: resize ────────────────────────────────────────────────────────────

def prepare_data(synthetic_model_dir: Path, image_size: int):
    '''Resize real and synthetic COCO images to image_size for training.'''
    training_dir = DATA_DIR / f'training_{image_size}'
    real_dst     = training_dir / 'real'
    synth_dst    = training_dir / 'synthetic'

    print(f'\n[Prepare] Resizing images to {image_size}×{image_size} …')
    for src, dst, label in [
        (DATA_DIR / 'coco_samples',         real_dst,  'real'),
        (synthetic_model_dir / 'coco',      synth_dst, 'synthetic'),
    ]:
        if not src.exists():
            print(f'  Source not found, skipping: {src}')
            continue
        _run_script('scripts/resize_for_training.py',
                    ['--src', str(src), '--dst', str(dst), '--size', str(image_size)])

    print(f'\nData ready in {training_dir}')
    print('Next step — upload to Lambda and train:')
    print(f'  rsync -avz {training_dir}/ ubuntu@<ip>:~/data/training_{image_size}/')
    print(f'  rsync -avz scripts/train_ddpm.py ubuntu@<ip>:~/scripts/')
    print()
    print('On Lambda, run:')
    print(f'  python scripts/train_ddpm.py --data-dir data/training_{image_size}/real '
          f'--output-dir checkpoints/ddpm_real --run-name ddpm-real --no-tracking')
    print(f'  python scripts/train_ddpm.py --data-dir data/training_{image_size}/synthetic '
          f'--output-dir checkpoints/ddpm_synthetic --run-name ddpm-synthetic --no-tracking')
    print()
    print('Then download checkpoints:')
    print('  rsync -avz ubuntu@<ip>:~/checkpoints/ ./checkpoints/')

    return real_dst, synth_dst


# ── step 2: train ─────────────────────────────────────────────────────────────

def run_training(image_size: int, num_epochs: int, batch_size: int, no_tracking: bool):
    '''Run both training jobs locally.'''
    training_dir = DATA_DIR / f'training_{image_size}'

    for label, data_subdir, ckpt_name in [
        ('real',      training_dir / 'real',      'ddpm_real'),
        ('synthetic', training_dir / 'synthetic',  'ddpm_synthetic'),
    ]:
        ckpt_dir = CHECKPOINTS / ckpt_name
        if ckpt_dir.exists() and (ckpt_dir / 'unet').exists():
            print(f'[Train] Checkpoint already exists for {label}, skipping: {ckpt_dir}')
            continue

        print(f'\n[Train] Training DDPM on {label} images …')
        extra = ['--no-tracking'] if no_tracking else []
        _run_script('scripts/train_ddpm.py', [
            '--data-dir',    str(data_subdir),
            '--output-dir',  str(ckpt_dir),
            '--image-size',  str(image_size),
            '--num-epochs',  str(num_epochs),
            '--batch-size',  str(batch_size),
            '--run-name',    f'ddpm-{label}',
            *extra,
        ])


# ── step 3: generate samples ──────────────────────────────────────────────────

def generate_samples(num_samples: int, batch_size: int):
    '''Generate samples from both DDPM checkpoints.'''
    samples = {}
    for label, ckpt_name in [('real', 'ddpm_real'), ('synthetic', 'ddpm_synthetic')]:
        ckpt_dir   = CHECKPOINTS / ckpt_name
        sample_dir = DATA_DIR / 'ddpm_samples' / label

        if not ckpt_dir.exists() or not (ckpt_dir / 'unet').exists():
            print(f'[Generate] Checkpoint not found for {label}: {ckpt_dir}')
            print('  Run training first, or download checkpoints from Lambda.')
            sys.exit(1)

        if sample_dir.exists() and len(list(sample_dir.glob('*.png'))) >= num_samples:
            print(f'[Generate] Samples already exist for {label}, skipping.')
        else:
            print(f'\n[Generate] Generating {num_samples} samples from {label} DDPM …')
            _run_script('scripts/generate_ddpm_samples.py', [
                '--checkpoint-dir', str(ckpt_dir),
                '--output-dir',     str(sample_dir),
                '--num-samples',    str(num_samples),
                '--batch-size',     str(batch_size),
            ])

        samples[label] = sample_dir
    return samples


# ── step 4: evaluate ──────────────────────────────────────────────────────────

def evaluate(samples: dict, logger):
    '''Compute FID for both sample sets vs real COCO images.'''
    real_ref = DATA_DIR / 'coco_samples'
    results  = {}

    for label, sample_dir in samples.items():
        print(f'\n[Evaluate] FID — DDPM trained on {label} images …')
        fid = evaluate_fid(
            real_dir=real_ref,
            gen_dir=sample_dir,
            logger=logger,
            log_key=f'fid_ddpm_{label}',
        )
        results[label] = fid
        print(f'  FID: {fid:.4f}')

    return results


# ── step 5: report ────────────────────────────────────────────────────────────

def _save_comparison_chart(fid_results: dict, synthetic_model: str,
                           run_name: str, logger):
    '''Bar chart: FID of real-trained vs synthetic-trained DDPM.'''
    sns.set_theme(style='whitegrid', font_scale=1.05)

    labels = ['DDPM trained\non real images', f'DDPM trained\non {synthetic_model}\nsynthetics']
    values = [fid_results.get('real', 0), fid_results.get('synthetic', 0)]
    colors = ['#4c72b0', '#c44e52']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.45)
    ax.set_ylabel('FID score (lower = closer to real distribution)')
    ax.set_title(
        f'Model Collapse Experiment\n'
        f'Synthetic source: {synthetic_model}',
        fontsize=12, fontweight='bold'
    )

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    delta = values[1] - values[0]
    sign  = '+' if delta >= 0 else ''
    ax.annotate(
        f'Collapse penalty: {sign}{delta:.2f} FID',
        xy=(0.5, 0.92), xycoords='axes fraction',
        ha='center', fontsize=10,
        color='#c44e52' if delta > 0 else '#4c72b0',
    )

    plt.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f'{run_name}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Chart saved → {out_path}')

    if logger:
        logger.log_image('model_collapse_chart', out_path)

    return out_path


def _save_results_json(synthetic_model: str, fid_results: dict,
                       run_name: str, timestamp: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_name':        run_name,
        'model':           f'loop4-{synthetic_model}',
        'dataset':         'coco',
        'timestamp':       timestamp,
        'metrics': {
            'fid_ddpm_real':      fid_results.get('real'),
            'fid_ddpm_synthetic': fid_results.get('synthetic'),
            'collapse_delta':     (fid_results.get('synthetic', 0)
                                   - fid_results.get('real', 0)),
        },
    }
    out_path = RESULTS_DIR / f'{run_name}.json'
    out_path.write_text(json.dumps(payload, indent=2))
    print(f'Results saved → {out_path}')


def _print_findings(fid_results: dict, synthetic_model: str):
    real_fid  = fid_results.get('real', float('nan'))
    synth_fid = fid_results.get('synthetic', float('nan'))
    delta     = synth_fid - real_fid

    print('\n── Loop 4 Findings ──────────────────────────────────────────')
    print(f'  FID (DDPM trained on real images)       : {real_fid:.2f}')
    print(f'  FID (DDPM trained on {synthetic_model:<14}): {synth_fid:.2f}')
    print(f'  Collapse penalty (Δ FID)                : {delta:+.2f}')
    print()
    if delta > 20:
        print('  → Strong model collapse signal: training on synthetic data')
        print('    substantially degraded the output distribution.')
    elif delta > 5:
        print('  → Moderate collapse signal: synthetic training has a')
        print('    measurable but limited effect on output quality.')
    else:
        print('  → Minimal collapse signal: the DDPM trained on synthetic')
        print('    data performs comparably to the real-data baseline.')


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_script(script: str, args: list):
    cmd = [sys.executable, str(PROJECT_ROOT / script), *args]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'Script failed: {" ".join(cmd)}')
        sys.exit(result.returncode)


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description='Loop 4 — model collapse experiment')
    parser.add_argument('--synthetic-model', required=True, choices=MODELS,
                        help='Which model\'s generated images to train the synthetic DDPM on')
    parser.add_argument('--image-size',  type=int, default=128,
                        help='Training resolution (default: 128)')
    parser.add_argument('--num-epochs',  type=int, default=200,
                        help='DDPM training epochs (default: 200)')
    parser.add_argument('--batch-size',  type=int, default=16,
                        help='Training and generation batch size (default: 16)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Images to generate per DDPM for FID (default: 1000)')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Resize images and print Lambda upload instructions, then exit')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (assumes checkpoints already exist in ./checkpoints/)')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip generation (assumes samples already exist in data/ddpm_samples/)')
    parser.add_argument('--run-name',    default=None,
                        help='W&B / MLflow run name')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable W&B and MLflow logging')
    return parser.parse_args()


def main():
    args          = _parse_args()
    timestamp     = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name      = args.run_name or f'loop4-{args.synthetic_model}--{timestamp}'
    synthetic_dir = DATA_DIR / 'generated_images' / MODELS[args.synthetic_model]

    # ── step 1: resize ────────────────────────────────────────────────────────
    if not args.skip_training:
        prepare_data(synthetic_dir, args.image_size)

    if args.prepare_only:
        return

    # ── step 2: train ─────────────────────────────────────────────────────────
    if not args.skip_training:
        run_training(args.image_size, args.num_epochs, args.batch_size, args.no_tracking)

    # ── step 3: generate samples ──────────────────────────────────────────────
    if not args.skip_generation:
        samples = generate_samples(args.num_samples, args.batch_size)
    else:
        samples = {
            'real':      DATA_DIR / 'ddpm_samples' / 'real',
            'synthetic': DATA_DIR / 'ddpm_samples' / 'synthetic',
        }

    # ── step 4 & 5: evaluate + report ────────────────────────────────────────
    logger = None
    if not args.no_tracking:
        from tracking.experiment_logger import DualLogger
        logger = DualLogger(run_name=run_name, config=vars(args))

    fid_results = evaluate(samples, logger)
    _save_comparison_chart(fid_results, args.synthetic_model, run_name, logger)
    _save_results_json(args.synthetic_model, fid_results, run_name, timestamp)
    _print_findings(fid_results, args.synthetic_model)

    if logger:
        logger.finish()
    print('\nDone.')


if __name__ == '__main__':
    main()
