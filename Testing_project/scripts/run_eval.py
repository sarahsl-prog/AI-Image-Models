'''
Main evaluation entrypoint.
Runs all metrics for one model against one or both datasets and logs results
to W&B and MLflow via DualLogger.

Usage:
    python scripts/run_eval.py --model flux-dev --dataset coco
    python scripts/run_eval.py --model flux-dev --dataset all
    python scripts/run_eval.py --model flux-dev --dataset imagenet --no-tracking
'''

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.clip_score import evaluate_clip
from evaluation.fid import evaluate_fid
from evaluation.inception_score import evaluate_inception_score
from evaluation.per_class import evaluate_per_class

# Maps short CLI names to the directory names used in data/generated_images/
MODELS = {
    'flux-dev':   'black-forest-labs--FLUX.1-dev',
    'flux-schnell': 'black-forest-labs--FLUX.1-schnell',
    'sd15':       'runwayml--stable-diffusion-v1-5',
    'sdxl':       'stabilityai--stable-diffusion-xl-base-1.0',
}

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'


def _parse_args():
    parser = argparse.ArgumentParser(description='Run image generation evaluation metrics')
    parser.add_argument('--model', required=True, choices=MODELS,
                        help='Model to evaluate')
    parser.add_argument('--dataset', default='all', choices=['coco', 'imagenet', 'all'],
                        help='Which dataset track to evaluate (default: all)')
    parser.add_argument('--run-name', default=None,
                        help='W&B / MLflow run name (defaults to model--dataset)')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable W&B and MLflow logging (useful for local testing)')
    return parser.parse_args()


def main():
    args = _parse_args()

    model_dir = DATA_DIR / 'generated_images' / MODELS[args.model]
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name  = args.run_name or f'{args.model}--{args.dataset}--{timestamp}'

    logger = None
    if not args.no_tracking:
        from tracking.experiment_logger import DualLogger
        logger = DualLogger(run_name=run_name, config=vars(args))

    # ── COCO track ────────────────────────────────────────────────────────────
    if args.dataset in ('coco', 'all'):
        print(f'\n[COCO] FID — {args.model}')
        fid = evaluate_fid(
            real_dir=DATA_DIR / 'coco_samples',
            gen_dir=model_dir / 'coco',
            logger=logger,
            log_key='fid_coco',
        )
        print(f'  FID: {fid:.4f}')

        print(f'[COCO] CLIP Score — {args.model}')
        clip = evaluate_clip(
            gen_dir=model_dir / 'coco',
            prompts_path=DATA_DIR / 'coco_prompts.json',
            logger=logger,
        )
        print(f'  CLIP Score: {clip:.4f}')

    # ── ImageNet track ────────────────────────────────────────────────────────
    if args.dataset in ('imagenet', 'all'):
        print(f'\n[ImageNet] FID — {args.model}')
        fid = evaluate_fid(
            real_dir=DATA_DIR / 'imagenet_samples',
            gen_dir=model_dir / 'imagenet',
            logger=logger,
            log_key='fid_imagenet',
        )
        print(f'  FID: {fid:.4f}')

        print(f'[ImageNet] Inception Score — {args.model}')
        is_mean, is_std = evaluate_inception_score(
            gen_dir=model_dir / 'imagenet',
            logger=logger,
        )
        print(f'  IS: {is_mean:.4f} ± {is_std:.4f}')

        print(f'[ImageNet] Per-Class FID — {args.model}')
        per_class = evaluate_per_class(
            real_dir=DATA_DIR / 'imagenet_samples',
            gen_dir=model_dir / 'imagenet',
            logger=logger,
        )
        for cls, score in sorted(per_class.items(), key=lambda x: x[1]):
            print(f'  {cls}: {score:.4f}')

    if logger:
        logger.finish()
    print('\nDone.')


if __name__ == '__main__':
    main()
