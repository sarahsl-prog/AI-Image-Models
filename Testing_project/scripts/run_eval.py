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
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
REPORT_DIR   = PROJECT_ROOT / 'report'


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


# ── results persistence ────────────────────────────────────────────────────────

def _save_results_json(model, dataset, run_name, timestamp, results):
    '''Saves metrics dict to report/results/{run_name}.json for use by plot_results.py.'''
    out_dir = REPORT_DIR / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_name':  run_name,
        'model':     model,
        'dataset':   dataset,
        'timestamp': timestamp,
        'metrics':   results,
    }
    out_path = out_dir / f'{run_name}.json'
    out_path.write_text(json.dumps(payload, indent=2))
    print(f'Results saved → {out_path}')
    return out_path


# ── summary chart ─────────────────────────────────────────────────────────────

# Metric display config: (key, label, direction_note, bar_color)
_METRIC_INFO = [
    ('fid_coco',     'FID — COCO',       'lower is better',  '#e05c5c'),
    ('clip_score',   'CLIP Score',        'higher is better', '#5c9ee0'),
    ('fid_imagenet', 'FID — ImageNet',    'lower is better',  '#e05c5c'),
    ('is_mean',      'Inception Score',   'higher is better', '#5c9ee0'),
]


def _save_summary_chart(model, results, run_name, logger):
    '''
    Generates a per-model bar chart for all computed metrics and saves it to
    report/runs/{run_name}.png. Logs to W&B + MLflow if logger is provided.
    '''
    sns.set_theme(style='whitegrid', font_scale=1.05)

    available = [(k, lbl, note, color)
                 for k, lbl, note, color in _METRIC_INFO
                 if k in results]
    if not available:
        return None

    n     = len(available)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    # Normalise axes to a flat list regardless of shape
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.flatten()) if hasattr(axes, 'flatten') else list(axes)

    for ax, (key, label, note, color) in zip(axes, available):
        val  = results[key]
        yerr = results.get('is_std') if key == 'is_mean' else None

        sns.barplot(x=[model], y=[val], ax=ax, color=color, width=0.4)
        if yerr is not None:
            ax.errorbar(x=0, y=val, yerr=yerr, fmt='none',
                        color='black', capsize=6, linewidth=2)

        ax.set_title(f'{label}\n({note})', fontsize=11)
        ax.set_ylabel(label)
        ax.set_xlabel('')
        ax.set_ylim(bottom=0)
        ax.text(0, val, f'  {val:.4f}', va='bottom', ha='center', fontsize=10)

    for ax in axes[n:]:   # hide any unused subplot cells
        ax.set_visible(False)

    fig.suptitle(f'Evaluation Summary — {model}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_dir = REPORT_DIR / 'runs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{run_name}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Summary chart saved → {out_path}')

    if logger:
        logger.log_image('summary_chart', out_path)

    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args      = _parse_args()
    model_dir = DATA_DIR / 'generated_images' / MODELS[args.model]
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name  = args.run_name or f'{args.model}--{args.dataset}--{timestamp}'

    logger = None
    if not args.no_tracking:
        from tracking.experiment_logger import DualLogger
        logger = DualLogger(run_name=run_name, config=vars(args))

    results = {}

    # ── COCO track ────────────────────────────────────────────────────────────
    if args.dataset in ('coco', 'all'):
        print(f'\n[COCO] FID — {args.model}')
        fid = evaluate_fid(
            real_dir=DATA_DIR / 'coco_samples',
            gen_dir=model_dir / 'coco',
            logger=logger,
            log_key='fid_coco',
        )
        results['fid_coco'] = fid
        print(f'  FID: {fid:.4f}')

        print(f'[COCO] CLIP Score — {args.model}')
        clip = evaluate_clip(
            gen_dir=model_dir / 'coco',
            prompts_path=DATA_DIR / 'coco_prompts.json',
            logger=logger,
        )
        results['clip_score'] = clip
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
        results['fid_imagenet'] = fid
        print(f'  FID: {fid:.4f}')

        print(f'[ImageNet] Inception Score — {args.model}')
        is_mean, is_std = evaluate_inception_score(
            gen_dir=model_dir / 'imagenet',
            logger=logger,
        )
        results['is_mean'] = is_mean
        results['is_std']  = is_std
        print(f'  IS: {is_mean:.4f} ± {is_std:.4f}')

        print(f'[ImageNet] Per-Class FID — {args.model}')
        per_class = evaluate_per_class(
            real_dir=DATA_DIR / 'imagenet_samples',
            gen_dir=model_dir / 'imagenet',
            logger=logger,
        )
        for cls, score in sorted(per_class.items(), key=lambda x: x[1]):
            print(f'  {cls}: {score:.4f}')

    # ── persist results and generate chart ────────────────────────────────────
    _save_results_json(args.model, args.dataset, run_name, timestamp, results)
    _save_summary_chart(args.model, results, run_name, logger)

    if logger:
        logger.finish()
    print('\nDone.')


if __name__ == '__main__':
    main()
