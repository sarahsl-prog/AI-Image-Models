'''
Inception Score (IS)
What it measures: Both quality (sharpness of class predictions) and diversity of generated images
Higher is better
Limitation: Biased toward ImageNet-like content; more meaningful for the ImageNet track
Computed for: Each model × ImageNet track only = 4 IS scores
'''

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights

from evaluation.fid import _ImageFolder


# ── private helpers ───────────────────────────────────────────────────────────

def _load_inception_classifier():
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.eval()
    return model


def _get_predictions(image_dir, model, batch_size=64):
    '''Returns softmax class probabilities, shape (N, 1000).'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loader = DataLoader(_ImageFolder(image_dir), batch_size=batch_size, num_workers=4)
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _compute_is(preds, splits=10):
    '''
    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )
    Averaged over `splits` random splits for a mean ± std estimate.
    '''
    n = preds.shape[0]
    split_scores = []
    for k in range(splits):
        part = preds[k * (n // splits): (k + 1) * (n // splits)]
        marginal = part.mean(axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(marginal + 1e-10))
        split_scores.append(np.exp(kl.sum(axis=1).mean()))
    return float(np.mean(split_scores)), float(np.std(split_scores))


# ── public API ────────────────────────────────────────────────────────────────

def evaluate_inception_score(gen_dir, logger=None, splits=10):
    '''
    Compute Inception Score for images in gen_dir.
    Returns (mean_is, std_is) — higher mean is better.
    Logs to logger if provided.
    '''
    model = _load_inception_classifier()
    preds = _get_predictions(gen_dir, model)
    mean_is, std_is = _compute_is(preds, splits=splits)
    if logger:
        logger.log({'inception_score_mean': mean_is, 'inception_score_std': std_is})
    return mean_is, std_is
