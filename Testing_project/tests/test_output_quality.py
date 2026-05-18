'''
Formal test suite for image generation model output quality.

Tests:
  test_fid_below_threshold      — at least one model achieves FID < threshold
                                  on the COCO track, confirming the pipeline
                                  produces meaningful generations
  test_clip_score_above_chance  — CLIP scores for matched image-caption pairs
                                  are statistically higher than random pairings

Run:
    pytest tests/test_output_quality.py
    pytest tests/test_output_quality.py -v
'''

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
RESULTS_DIR  = PROJECT_ROOT / 'report' / 'results'

# Models in preference order for selecting sample images
_MODEL_DIRS = [
    'black-forest-labs--FLUX.1-dev',
    'stabilityai--stable-diffusion-xl-base-1.0',
    'black-forest-labs--FLUX.1-schnell',
    'runwayml--stable-diffusion-v1-5',
]

# ── thresholds ────────────────────────────────────────────────────────────────

# At least one evaluated model must beat this FID on the COCO track.
# State-of-the-art models typically score 20–80; 150 is a conservative bar
# that rules out noise or degenerate outputs while remaining achievable.
FID_COCO_THRESHOLD = 150.0

# Matched (image, caption) CLIP scores must exceed shuffled (random) pairs
# by at least this margin. Guards against the matched mean winning by noise alone.
CLIP_MIN_DELTA = 0.01

# Number of image-caption pairs sampled for the CLIP chance comparison.
CLIP_SAMPLE_SIZE = 100


# ── shared helpers ────────────────────────────────────────────────────────────

def _load_results():
    '''Return list of all result dicts from report/results/*.json.'''
    if not RESULTS_DIR.exists():
        return []
    records = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        try:
            records.append(json.loads(path.read_text()))
        except Exception:
            continue
    return records


def _latest_per_model(records):
    '''Return {model: result_dict} keeping only the latest run per model.'''
    latest = {}
    for r in sorted(records, key=lambda x: x.get('timestamp', '')):
        latest[r['model']] = r
    return latest


def _best_coco_gen_dir():
    '''
    Return the Path to the COCO generated-image directory for the first model
    (in preference order) that has images available, or None.
    '''
    for model_dir in _MODEL_DIRS:
        path = DATA_DIR / 'generated_images' / model_dir / 'coco'
        images = list(path.glob('*.png')) + list(path.glob('*.jpg'))
        if images:
            return path
    return None


def _clip_mean(image_paths, captions, model, processor, device, batch_size=50):
    '''Return mean cosine similarity between each image and its paired caption.'''
    scores = []
    for i in range(0, len(image_paths), batch_size):
        imgs = [Image.open(p).convert('RGB') for p in image_paths[i:i + batch_size]]
        caps = captions[i:i + batch_size]
        inputs = processor(text=caps, images=imgs, return_tensors='pt',
                           padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        scores.append((img_emb * txt_emb).sum(dim=-1).cpu().numpy())
    return float(np.concatenate(scores).mean())


# ── tests ─────────────────────────────────────────────────────────────────────

def test_fid_below_threshold():
    '''
    Assert that at least one model achieves FID < FID_COCO_THRESHOLD on the
    COCO validation set, confirming the evaluation pipeline produces
    meaningful results rather than noise or degenerate images.

    Reads from pre-computed results in report/results/. Skips if no results
    are available (run scripts/run_eval.py first).
    '''
    records = _load_results()
    if not records:
        pytest.skip('No evaluation results found — run scripts/run_eval.py first')

    coco_fids = {
        model: r['metrics']['fid_coco']
        for model, r in _latest_per_model(records).items()
        if 'fid_coco' in r.get('metrics', {})
    }

    if not coco_fids:
        pytest.skip('No fid_coco metrics available — run run_eval.py with --dataset coco or all')

    best_model = min(coco_fids, key=coco_fids.get)
    best_fid   = coco_fids[best_model]

    assert best_fid < FID_COCO_THRESHOLD, (
        f'No model achieved FID < {FID_COCO_THRESHOLD} on COCO. '
        f'Best: {best_model} (FID={best_fid:.2f}). '
        f'All scores: { {m: f"{v:.2f}" for m, v in coco_fids.items()} }'
    )


def test_clip_score_above_chance():
    '''
    Assert that CLIP scores for generated images are statistically higher
    than random image-caption pairings.

    Samples CLIP_SAMPLE_SIZE matched (image, caption) pairs from the best
    available model's COCO outputs, computes the mean CLIP score, then
    shuffles captions to produce a random-pairing baseline. Asserts that
    the matched mean exceeds the shuffled mean by at least CLIP_MIN_DELTA.

    Skips if generated images or COCO prompts are not available.
    '''
    gen_dir = _best_coco_gen_dir()
    if gen_dir is None:
        pytest.skip('No generated COCO images found — populate data/generated_images/ first')

    prompts_path = DATA_DIR / 'coco_prompts.json'
    if not prompts_path.exists():
        pytest.skip('coco_prompts.json not found — run scripts/coco_scripts/recover_coco_prompts.py first')

    # Load prompts
    prompts = {e['index']: e['caption']
               for e in json.loads(prompts_path.read_text())}

    # Collect matched (image_path, caption) pairs, capped at CLIP_SAMPLE_SIZE
    image_paths = sorted(
        p for p in gen_dir.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )[:CLIP_SAMPLE_SIZE]

    if len(image_paths) < 10:
        pytest.skip(f'Too few images in {gen_dir} to run CLIP chance test (found {len(image_paths)})')

    matched_captions  = [prompts[int(p.stem)] for p in image_paths]
    shuffled_captions = matched_captions.copy()
    rng = np.random.default_rng(seed=42)
    rng.shuffle(shuffled_captions)

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id  = 'openai/clip-vit-base-patch32'
    clip      = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    matched_mean  = _clip_mean(image_paths, matched_captions,  clip, processor, device)
    shuffled_mean = _clip_mean(image_paths, shuffled_captions, clip, processor, device)

    assert matched_mean > shuffled_mean + CLIP_MIN_DELTA, (
        f'Matched CLIP score ({matched_mean:.4f}) did not exceed shuffled baseline '
        f'({shuffled_mean:.4f}) by the required margin of {CLIP_MIN_DELTA}. '
        f'Delta: {matched_mean - shuffled_mean:.4f}'
    )
