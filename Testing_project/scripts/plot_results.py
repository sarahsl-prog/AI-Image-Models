'''
Cross-model comparison charts.

Reads all per-run JSON files written by run_eval.py from report/results/,
selects the latest run per (model, dataset) combination, and generates four
seaborn comparison charts saved to report/comparison_charts/:

    fid_comparison.png       — FID scores, COCO and ImageNet side-by-side
    clip_comparison.png      — CLIP scores (COCO track)
    is_comparison.png        — Inception Score mean ± std (ImageNet track)
    overview.png             — All three charts in a single figure

Optionally logs all charts to W&B and MLflow as artifacts.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results-dir report/results --no-tracking
    python scripts/plot_results.py --run-name comparison-2026-03-22
'''

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_DIR   = PROJECT_ROOT / 'report'

# Consistent color palette — one colour per model, used across all charts
MODEL_PALETTE = {
    'flux-dev':     '#4c72b0',
    'flux-schnell': '#55a868',
    'sd15':         '#c44e52',
    'sdxl':         '#dd8452',
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load_results(results_dir: Path) -> pd.DataFrame:
    '''
    Reads all JSON files from results_dir, keeps the latest run per
    (model, dataset), and returns a tidy DataFrame with one row per
    (model, metric, value).
    '''
    if not results_dir.exists():
        return None

    records = []
    for path in sorted(results_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        row = {
            'run_name':  data.get('run_name', path.stem),
            'model':     data.get('model', ''),
            'dataset':   data.get('dataset', ''),
            'timestamp': data.get('timestamp', ''),
        }
        row.update(data.get('metrics', {}))
        records.append(row)

    if not records:
        return None

    df = pd.DataFrame(records)

    # Keep the latest run per (model, dataset)
    df = (df.sort_values('timestamp')
            .drop_duplicates(subset=['model', 'dataset'], keep='last')
            .reset_index(drop=True))
    return df


# ── individual chart builders ─────────────────────────────────────────────────

def _model_order(df):
    '''Return models sorted by their first appearance in MODEL_PALETTE, then alphabetically.'''
    palette_order = list(MODEL_PALETTE.keys())
    present = df['model'].unique().tolist()
    ordered = [m for m in palette_order if m in present]
    ordered += sorted(m for m in present if m not in palette_order)
    return ordered


def _palette_for(models):
    return [MODEL_PALETTE.get(m, '#888888') for m in models]


def _fid_chart(df, ax_coco, ax_imagenet):
    '''
    Two side-by-side bar charts: FID on COCO and FID on ImageNet.
    Rows with the relevant metric missing are silently skipped.
    '''
    models = _model_order(df)

    for ax, col, title in [
        (ax_coco,      'fid_coco',      'FID — COCO\n(lower is better)'),
        (ax_imagenet,  'fid_imagenet',  'FID — ImageNet\n(lower is better)'),
    ]:
        sub = df[df[col].notna()].copy()
        sub = sub.set_index('model').reindex([m for m in models if m in sub.index])
        if sub.empty:
            ax.set_visible(False)
            continue

        palette = _palette_for(sub.index.tolist())
        sns.barplot(x=sub.index.tolist(), y=sub[col].tolist(),
                    palette=palette, ax=ax, width=0.55)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('FID score')
        ax.tick_params(axis='x', rotation=15)

        for bar, val in zip(ax.patches, sub[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)


def _clip_chart(df, ax):
    '''Bar chart of CLIP scores across models.'''
    models = _model_order(df)
    sub = df[df['clip_score'].notna()].copy()
    sub = sub.set_index('model').reindex([m for m in models if m in sub.index])
    if sub.empty:
        ax.set_visible(False)
        return

    palette = _palette_for(sub.index.tolist())
    sns.barplot(x=sub.index.tolist(), y=sub['clip_score'].tolist(),
                palette=palette, ax=ax, width=0.55)
    ax.set_title('CLIP Score — COCO\n(higher is better)', fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('Mean cosine similarity')
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylim(bottom=0)

    for bar, val in zip(ax.patches, sub['clip_score']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)


def _is_chart(df, ax):
    '''Bar chart of Inception Score mean with ± std error bars.'''
    models = _model_order(df)
    sub = df[df['is_mean'].notna()].copy()
    sub = sub.set_index('model').reindex([m for m in models if m in sub.index])
    if sub.empty:
        ax.set_visible(False)
        return

    palette = _palette_for(sub.index.tolist())
    means  = sub['is_mean'].tolist()
    stds   = sub['is_std'].fillna(0).tolist() if 'is_std' in sub.columns else [0] * len(means)

    bars = ax.bar(sub.index.tolist(), means, color=palette, width=0.55)
    ax.errorbar(x=range(len(means)), y=means, yerr=stds,
                fmt='none', color='black', capsize=6, linewidth=1.5)
    ax.set_title('Inception Score — ImageNet\n(higher is better)', fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('IS mean')
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylim(bottom=0)

    for bar, val, std in zip(bars, means, stds):
        label = f'{val:.2f} ± {std:.2f}' if std else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds or [0]) + 0.1,
                label, ha='center', va='bottom', fontsize=9)


# ── chart savers ──────────────────────────────────────────────────────────────

def _save(fig, path, logger, log_key):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Chart saved → {path}')
    if logger:
        logger.log_image(log_key, path)


def generate_charts(df, output_dir: Path, logger=None):
    '''
    Generates and saves all comparison charts. Returns list of output paths.
    '''
    sns.set_theme(style='whitegrid', font_scale=1.05)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # ── FID comparison ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('FID Comparison Across Models', fontsize=13, fontweight='bold')
    _fid_chart(df, ax1, ax2)
    path = output_dir / 'fid_comparison.png'
    _save(fig, path, logger, 'chart/fid_comparison')
    saved.append(path)

    # ── CLIP score ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('CLIP Score Comparison', fontsize=13, fontweight='bold')
    _clip_chart(df, ax)
    path = output_dir / 'clip_comparison.png'
    _save(fig, path, logger, 'chart/clip_comparison')
    saved.append(path)

    # ── Inception Score ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Inception Score Comparison', fontsize=13, fontweight='bold')
    _is_chart(df, ax)
    path = output_dir / 'is_comparison.png'
    _save(fig, path, logger, 'chart/is_comparison')
    saved.append(path)

    # ── overview: all four subplots in one figure ─────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Model Evaluation Overview', fontsize=14, fontweight='bold')
    ax_fid_coco = fig.add_subplot(2, 2, 1)
    ax_fid_in   = fig.add_subplot(2, 2, 2)
    ax_clip     = fig.add_subplot(2, 2, 3)
    ax_is       = fig.add_subplot(2, 2, 4)
    _fid_chart(df, ax_fid_coco, ax_fid_in)
    _clip_chart(df, ax_clip)
    _is_chart(df, ax_is)
    path = output_dir / 'overview.png'
    _save(fig, path, logger, 'chart/overview')
    saved.append(path)

    return saved


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description='Generate cross-model comparison charts')
    parser.add_argument('--results-dir', default=str(REPORT_DIR / 'results'),
                        help='Directory containing per-run result JSON files')
    parser.add_argument('--output-dir', default=str(REPORT_DIR / 'comparison_charts'),
                        help='Where to save the comparison charts')
    parser.add_argument('--run-name', default=None,
                        help='W&B / MLflow run name (defaults to plot-results--<timestamp>)')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable W&B and MLflow logging')
    return parser.parse_args()


def main():
    args        = _parse_args()
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    print(f'Loading results from {results_dir} …')
    df = _load_results(results_dir)
    if df is None:
        print(f'No result JSON files found in {results_dir}.')
        print('Run scripts/run_eval.py for at least one model first.')
        return
    print(f'  {len(df)} runs loaded:')
    for _, row in df.iterrows():
        print(f'    {row["model"]:20}  dataset={row["dataset"]:10}  ts={row["timestamp"]}')

    logger = None
    if not args.no_tracking:
        from tracking.experiment_logger import DualLogger
        run_name = args.run_name or f'plot-results--{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        logger   = DualLogger(run_name=run_name, config=vars(args))

    saved = generate_charts(df, output_dir, logger)

    if logger:
        logger.finish()

    print(f'\nDone. {len(saved)} charts saved to {output_dir}')


if __name__ == '__main__':
    main()
