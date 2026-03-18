'''
Fréchet Inception Distance (FID)
What it measures: Distribution-level similarity between generated and real
images using InceptionV3 feature statistics
Lower is better
Computed for: Each model × each dataset (COCO and ImageNet) = 8 FID scores
'''

import numpy as np
import torch
from pathlib import Path
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import InterpolationMode


# ── private helpers ───────────────────────────────────────────────────────────

def _load_inception_model():
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()   # return 2048-d pool features, not logits
    model.eval()
    return model


_transform = transforms.Compose([
    transforms.Resize((299, 299), interpolation=InterpolationMode.BILINEAR),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _ImageFolder(Dataset):
    def __init__(self, image_dir):
        self.paths = sorted(Path(image_dir).rglob('*'))
        self.paths = [p for p in self.paths if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_image(str(self.paths[idx]))
        if img.shape[0] == 1:           # grayscale → RGB
            img = img.expand(3, -1, -1)
        if img.shape[0] == 4:           # RGBA → RGB
            img = img[:3]
        return _transform(img)


def _extract_features(image_dir, model, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loader = DataLoader(_ImageFolder(image_dir), batch_size=batch_size, num_workers=4)
    features = []
    with torch.no_grad():
        for batch in loader:
            features.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(features, axis=0)


def _compute_statistics(features):
    mu = np.mean(features, axis=0)
    ddof = min(1, features.shape[0] - 1)   # np.cov requires ddof < N; guard for N=1
    sigma = np.cov(features, rowvar=False, ddof=ddof)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    '''Fréchet distance between two multivariate Gaussians.'''
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():          # numerical fix
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):                # drop negligible imaginary part
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# ── public API ────────────────────────────────────────────────────────────────

def evaluate_fid(real_dir, gen_dir, logger=None, log_key='fid'):
    '''
    Compute FID between images in real_dir and gen_dir.
    Returns FID score (float, lower is better).
    Logs to logger if provided.
    '''
    model = _load_inception_model()
    real_features = _extract_features(real_dir, model)
    gen_features  = _extract_features(gen_dir,  model)
    mu_r, sigma_r = _compute_statistics(real_features)
    mu_g, sigma_g = _compute_statistics(gen_features)
    score = _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    if logger:
        logger.log({log_key: score})
    return score
