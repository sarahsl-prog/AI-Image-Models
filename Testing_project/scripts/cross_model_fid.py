'''
Loop 3 — Cross-Model Distribution Analysis

Computes pairwise FID between every pair of model output distributions
(not vs. real images) for both COCO and ImageNet tracks.

Key question: Does FLUX.1-schnell distillation meaningfully alter the output
distribution relative to FLUX.1-dev?

Output:
  - 4×4 FID matrix printed to stdout for each dataset
  - Heatmap saved to report/pairwise_fid/
  - All scores logged to W&B + MLflow

Usage:
    python scripts/cross_model_fid.py
    python scripts/cross_model_fid.py --dataset coco --no-tracking
'''

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.fid import evaluate_fid

PROJECT_ROOT   = Path(__file__).parent.parent
DATA_DIR       = PROJECT_ROOT / 'data'
GENERATED_ROOT = DATA_DIR / 'generated_images'

MODELS = {
    'flux-dev':     'black-forest-labs--FLUX.1-dev',
    'flux-schnell': 'black-forest-labs--FLUX.1-schnell',
    'sd15':         'runwayml--stable-diffusion-v1-5',
    'sdxl':         'stabilityai--stable-diffusion-xl-base-1.0',
}
MODEL_KEYS = list(MODELS.keys())


# ── pairwise FID ──────────────────────────────────────────────────────────────

def _compute_matrix(dataset, logger):
    '''
    Returns a symmetric (N×N) FID matrix over MODEL_KEYS.
    Diagonal is 0 by definition. Only the upper triangle is computed;
    lower triangle is filled by symmetry (FID is symmetric).
    '''
    n = len(MODEL_KEYS)
    matrix = np.zeros((n, n))

    pairs = list(itertools.combinations(range(n), 2))
    total = len(pairs)

    for step, (i, j) in enumerate(pairs, 1):
        key_a, key_b = MODEL_KEYS[i], MODEL_KEYS[j]
        dir_a = GENERATED_ROOT / MODELS[key_a] / dataset
        dir_b = GENERATED_ROOT / MODELS[key_b] / dataset
        log_key = f'pairwise_fid_{dataset}_{key_a}_vs_{key_b}'

        print(f'  [{step}/{total}] {key_a} vs {key_b}')
        score = evaluate_fid(dir_a, dir_b, logger=logger, log_key=log_key)
        matrix[i, j] = score
        matrix[j, i] = score   # symmetric

    return matrix


# ── visualization ─────────────────────────────────────────────────────────────

def _save_heatmap(matrix, dataset, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='YlOrRd')

    ax.set_xticks(range(len(MODEL_KEYS)))
    ax.set_yticks(range(len(MODEL_KEYS)))
    ax.set_xticklabels(MODEL_KEYS, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(MODEL_KEYS, fontsize=10)

    # annotate cells
    for i in range(len(MODEL_KEYS)):
        for j in range(len(MODEL_KEYS)):
            val = matrix[i, j]
            text = '0' if i == j else f'{val:.1f}'
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label='FID (lower = more similar)')
    ax.set_title(f'Pairwise FID — {dataset.upper()} track\n'
                 f'(lower = distributions are more alike)', fontsize=12)
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'pairwise_fid_{dataset}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Heatmap saved → {out_path}')
    plt.close()


# ── findings summary ──────────────────────────────────────────────────────────

def _print_findings(matrix, dataset):
    print(f'\n── Pairwise FID matrix ({dataset.upper()}) ──')
    col_w = 14
    header = f'{"":20}' + ''.join(f'{k:>{col_w}}' for k in MODEL_KEYS)
    print(header)
    for i, key in enumerate(MODEL_KEYS):
        row = f'{key:<20}' + ''.join(
            f'{"—":>{col_w}}' if i == j else f'{matrix[i,j]:>{col_w}.2f}'
            for j in range(len(MODEL_KEYS))
        )
        print(row)

    # highlight the flux-dev vs flux-schnell pair specifically
    i_dev     = MODEL_KEYS.index('flux-dev')
    i_schnell = MODEL_KEYS.index('flux-schnell')
    flux_fid  = matrix[i_dev, i_schnell]

    # find the closest and furthest other pair (excluding diagonal)
    n = len(MODEL_KEYS)
    off_diag = [(matrix[i, j], MODEL_KEYS[i], MODEL_KEYS[j])
                for i in range(n) for j in range(i + 1, n)]
    off_diag.sort()
    most_similar  = off_diag[0]
    most_different = off_diag[-1]

    print(f'\n── Key findings ({dataset.upper()}) ──')
    print(f'  flux-dev vs flux-schnell FID : {flux_fid:.2f}')
    print(f'  Most similar pair            : {most_similar[1]} vs {most_similar[2]}  '
          f'(FID={most_similar[0]:.2f})')
    print(f'  Most different pair          : {most_different[1]} vs {most_different[2]}  '
          f'(FID={most_different[0]:.2f})')
    if flux_fid == most_similar[0]:
        print('  → flux-dev and flux-schnell are the MOST similar pair — '
              'distillation preserves distribution well.')
    elif flux_fid < np.median([x[0] for x in off_diag]):
        print('  → flux-dev and flux-schnell are closer than average — '
              'distillation has moderate impact on distribution.')
    else:
        print('  → flux-dev and flux-schnell are further apart than average — '
              'distillation meaningfully shifts the output distribution.')


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description='Pairwise cross-model FID analysis')
    parser.add_argument('--dataset', default='all', choices=['coco', 'imagenet', 'all'])
    parser.add_argument('--output-dir', default='report/pairwise_fid')
    parser.add_argument('--no-tracking', action='store_true')
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    datasets = ['coco', 'imagenet'] if args.dataset == 'all' else [args.dataset]

    logger = None
    if not args.no_tracking:
        from datetime import datetime
        from tracking.experiment_logger import DualLogger
        run_name = f'pairwise-fid--{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        logger = DualLogger(run_name=run_name, config=vars(args))

    for dataset in datasets:
        print(f'\nComputing pairwise FID — {dataset.upper()} track '
              f'({len(MODEL_KEYS) * (len(MODEL_KEYS) - 1) // 2} pairs)…')
        matrix = _compute_matrix(dataset, logger)
        _save_heatmap(matrix, dataset, output_dir)
        _print_findings(matrix, dataset)

    if logger:
        logger.finish()
    print('\nDone.')


if __name__ == '__main__':
    main()
