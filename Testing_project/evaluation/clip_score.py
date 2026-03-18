'''
CLIP Score
What it measures: Semantic alignment between a generated image and its source caption
Higher is better
COCO track only — requires recovered captions from coco_prompts.json
Computed for: Each model's 2,500 COCO-generated images vs. their source captions
'''

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ── private helpers ───────────────────────────────────────────────────────────

def _load_clip_model(model_id='openai/clip-vit-base-patch32'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, device


def _load_prompts(prompts_path):
    with open(prompts_path) as f:
        entries = json.load(f)
    # entries are [{index, image_id, caption}, ...] sorted by index
    return {e['index']: e['caption'] for e in entries}


def _score_batch(image_paths, captions, model, processor, device):
    images = [Image.open(p).convert('RGB') for p in image_paths]
    inputs = processor(text=captions, images=images, return_tensors='pt',
                       padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # cosine similarity between each image and its paired caption
    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return (img_emb * txt_emb).sum(dim=-1).cpu().numpy()


# ── public API ────────────────────────────────────────────────────────────────

def evaluate_clip(gen_dir, prompts_path, logger=None, batch_size=64,
                  model_id='openai/clip-vit-base-patch32'):
    '''
    Compute mean CLIP score for images in gen_dir against captions in prompts_path.
    Images must be named 0000.png, 0001.png, ... matching prompt indices.
    Returns mean CLIP score (float, higher is better).
    Logs to logger if provided.
    '''
    model, processor, device = _load_clip_model(model_id)
    prompts = _load_prompts(prompts_path)

    gen_dir = Path(gen_dir)
    image_paths = sorted(gen_dir.glob('*'))
    image_paths = [p for p in image_paths if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]

    all_scores = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        # derive prompt index from filename stem (e.g. "0042.png" → 42)
        captions = [prompts[int(p.stem)] for p in batch_paths]
        scores = _score_batch(batch_paths, captions, model, processor, device)
        all_scores.append(scores)

    mean_score = float(np.concatenate(all_scores).mean())
    if logger:
        logger.log({'clip_score': mean_score})
    return mean_score
