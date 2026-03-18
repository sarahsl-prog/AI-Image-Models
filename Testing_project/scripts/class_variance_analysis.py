'''
Loop 2 — Per-Class Deep Dive (ImageNet)

For each overlapping class between real and generated images:
  - Compute mean CLIP score (image vs. class label text) per model as quality proxy
  - Identify the N classes with highest variance across models
  - Visualize one generated sample per model side-by-side for each top class
  - Compute per-class FID for those top classes (approximate — ~3 gen images each)

Note on FID: per-class FID is approximate given only 3 generated images per class.
It is useful for ordering but not for absolute interpretation.

Usage:
    python scripts/class_variance_analysis.py
    python scripts/class_variance_analysis.py --top-n 5 --no-tracking
    python scripts/class_variance_analysis.py --output-dir report/loop2
'''

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models import Inception_V3_Weights
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.fid import evaluate_fid

PROJECT_ROOT   = Path(__file__).parent.parent
DATA_DIR       = PROJECT_ROOT / 'data'
IMAGENET_REAL  = DATA_DIR / 'imagenet_samples'
GENERATED_ROOT = DATA_DIR / 'generated_images'

MODELS = {
    'flux-dev':     'black-forest-labs--FLUX.1-dev',
    'flux-schnell': 'black-forest-labs--FLUX.1-schnell',
    'sd15':         'runwayml--stable-diffusion-v1-5',
    'sdxl':         'stabilityai--stable-diffusion-xl-base-1.0',
}
MODEL_LABELS = list(MODELS.keys())


# ── data helpers ──────────────────────────────────────────────────────────────

def _build_label_map():
    '''Returns {label_id: class_name} from torchvision InceptionV3 categories.'''
    cats = Inception_V3_Weights.DEFAULT.meta['categories']
    return {i: name for i, name in enumerate(cats)}


def _group_real_by_class(label_map):
    '''
    Parses imagenet_samples filenames (e.g. 00000_label856.png) and groups
    paths by class name. Only keeps classes that exist as generated subdirs
    for ALL models.
    '''
    real_by_class = defaultdict(list)
    for path in IMAGENET_REAL.glob('*.png'):
        m = re.search(r'_label(\d+)', path.name)
        if m:
            name = label_map[int(m.group(1))]
            real_by_class[name].append(path)

    # keep only classes present in every model's generated dir
    shared = set(real_by_class.keys())
    for model_dir in MODELS.values():
        gen_imagenet = GENERATED_ROOT / model_dir / 'imagenet'
        shared &= {d.name for d in gen_imagenet.iterdir() if d.is_dir()}

    return {cls: real_by_class[cls] for cls in shared}


def _gen_images_for(model_key, class_name):
    model_dir = MODELS[model_key]
    return sorted((GENERATED_ROOT / model_dir / 'imagenet' / class_name).glob('*.png'))


# ── CLIP scoring ──────────────────────────────────────────────────────────────

def _clip_score_images(image_paths, text, model, processor, device):
    '''Mean cosine similarity between each image and text.'''
    images = [Image.open(p).convert('RGB') for p in image_paths]
    inputs = processor(text=[text] * len(images), images=images,
                       return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img * txt).sum(dim=-1).mean().cpu())


def _compute_class_clip_scores(shared_classes, clip_model, processor, device):
    '''
    Returns {class_name: {model_key: mean_clip_score}} for all shared classes.
    '''
    scores = {}
    total = len(shared_classes)
    for i, cls in enumerate(shared_classes, 1):
        print(f'  CLIP scoring [{i}/{total}] {cls}', end='\r')
        prompt = f'a photo of a {cls}'
        scores[cls] = {}
        for model_key in MODEL_LABELS:
            paths = _gen_images_for(model_key, cls)
            if paths:
                scores[cls][model_key] = _clip_score_images(paths, prompt,
                                                            clip_model, processor, device)
    print()
    return scores


# ── variance analysis ─────────────────────────────────────────────────────────

def _top_variance_classes(clip_scores, top_n):
    '''Returns list of (class_name, variance) sorted descending by variance.'''
    variances = []
    for cls, model_scores in clip_scores.items():
        if len(model_scores) == len(MODEL_LABELS):   # all models present
            vals = [model_scores[m] for m in MODEL_LABELS]
            variances.append((cls, float(np.var(vals))))
    return sorted(variances, key=lambda x: x[1], reverse=True)[:top_n]


# ── visualization ─────────────────────────────────────────────────────────────

def _make_grid(top_classes, output_path):
    '''
    Grid: rows = classes, columns = models.
    Shows one generated sample per cell.
    '''
    n_classes = len(top_classes)
    n_models  = len(MODEL_LABELS)
    fig, axes = plt.subplots(n_classes, n_models,
                             figsize=(4 * n_models, 4 * n_classes))

    for row, (cls, variance) in enumerate(top_classes):
        for col, model_key in enumerate(MODEL_LABELS):
            ax = axes[row][col] if n_classes > 1 else axes[col]
            paths = _gen_images_for(model_key, cls)
            if paths:
                ax.imshow(Image.open(paths[0]).convert('RGB'))
            ax.axis('off')
            if row == 0:
                ax.set_title(model_key, fontsize=11, fontweight='bold')
        # class label on the left of each row
        label_ax = axes[row][0] if n_classes > 1 else axes[0]
        label_ax.set_ylabel(f'{cls}\n(var={variance:.4f})', fontsize=9, rotation=0,
                            labelpad=80, va='center')

    plt.suptitle('Generated samples — top variance ImageNet classes', fontsize=14)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f'Grid saved → {output_path}')


# ── per-class FID ─────────────────────────────────────────────────────────────

def _compute_top_class_fid(top_classes, real_by_class, tmp_dir, logger):
    '''
    Writes real images for each top class to a temp dir and runs evaluate_fid.
    Returns {class_name: {model_key: fid_score}}.
    Note: approximate — very few images per class.
    '''
    import shutil
    import tempfile

    fid_scores = {}
    for cls, _ in top_classes:
        fid_scores[cls] = {}
        real_paths = real_by_class[cls]

        # write real images for this class to a temp dir
        real_tmp = Path(tempfile.mkdtemp(dir=tmp_dir))
        for p in real_paths:
            shutil.copy(p, real_tmp / p.name)

        for model_key in MODEL_LABELS:
            gen_dir = GENERATED_ROOT / MODELS[model_key] / 'imagenet' / cls
            log_key = f'fid_{cls.replace(" ", "_")}_{model_key}'
            try:
                score = evaluate_fid(real_tmp, gen_dir, logger=logger, log_key=log_key)
            except Exception as e:
                print(f'  FID skipped ({cls} / {model_key}): {e}')
                score = float('nan')
            fid_scores[cls][model_key] = score

        shutil.rmtree(real_tmp)

    return fid_scores


# ── report ────────────────────────────────────────────────────────────────────

def _print_report(top_classes, clip_scores, fid_scores):
    col_w = 14
    header = f'{"Class":<30}' + ''.join(f'{m:>{col_w}}' for m in MODEL_LABELS)
    print('\n── CLIP scores (quality proxy, higher = better) ──')
    print(header)
    for cls, var in top_classes:
        row = f'{cls:<30}' + ''.join(
            f'{clip_scores[cls].get(m, float("nan")):>{col_w}.4f}' for m in MODEL_LABELS)
        print(row)

    print('\n── Per-class FID (approximate, lower = better) ──')
    print(header)
    for cls, _ in top_classes:
        row = f'{cls:<30}' + ''.join(
            f'{fid_scores[cls].get(m, float("nan")):>{col_w}.4f}' for m in MODEL_LABELS)
        print(row)


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description='Per-class ImageNet variance analysis')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of high-variance classes to analyse (default: 10)')
    parser.add_argument('--output-dir', default='report/loop2',
                        help='Where to save the grid image and report')
    parser.add_argument('--clip-model', default='openai/clip-vit-base-patch32')
    parser.add_argument('--no-tracking', action='store_true')
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = PROJECT_ROOT / args.output_dir

    label_map = _build_label_map()
    print(f'Building class index…')
    real_by_class = _group_real_by_class(label_map)
    print(f'  {len(real_by_class)} classes shared across all models')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading CLIP ({args.clip_model})…')
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor  = CLIPProcessor.from_pretrained(args.clip_model)

    logger = None
    if not args.no_tracking:
        from datetime import datetime
        from tracking.experiment_logger import DualLogger
        run_name = f'loop2-class-variance--{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        logger = DualLogger(run_name=run_name, config=vars(args))

    print('Computing per-class CLIP scores…')
    clip_scores = _compute_class_clip_scores(real_by_class, clip_model, processor, device)

    print(f'Identifying top {args.top_n} classes by cross-model variance…')
    top_classes = _top_variance_classes(clip_scores, args.top_n)
    for cls, var in top_classes:
        print(f'  {cls:<35} variance={var:.5f}')

    print('Generating visualization grid…')
    _make_grid(top_classes, output_dir / 'class_variance_grid.png')

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        print('Computing per-class FID for top classes (approximate)…')
        fid_scores = _compute_top_class_fid(top_classes, real_by_class, tmp, logger)

    _print_report(top_classes, clip_scores, fid_scores)

    if logger:
        logger.finish()
    print('\nDone.')


if __name__ == '__main__':
    main()
