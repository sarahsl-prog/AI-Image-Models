import argparse
import random
from pathlib import Path

import torch
import numpy as np
import wandb
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fid_ss import calculate_frechet_distance


def calculate_kid(real_features, gen_features, num_subsets=100, subset_size=1000):
  d = real_features.shape[1]
  n_real, n_gen = len(real_features), len(gen_features)
  subset_size = min(subset_size, n_real, n_gen)
  rng = np.random.default_rng()
  kid_scores = []
  for _ in range(num_subsets):
    real_sub = real_features[rng.choice(n_real, subset_size, replace=False)]
    gen_sub = gen_features[rng.choice(n_gen, subset_size, replace=False)]
    def poly_kernel(X, Y):
      return ((X @ Y.T) / d + 1) ** 3
    kxx = poly_kernel(real_sub, real_sub)
    kyy = poly_kernel(gen_sub, gen_sub)
    kxy = poly_kernel(real_sub, gen_sub)
    n, m = len(real_sub), len(gen_sub)
    mmd = ((np.sum(kxx) - np.trace(kxx)) / (n * (n - 1))
           + (np.sum(kyy) - np.trace(kyy)) / (m * (m - 1))
           - 2 * np.mean(kxy))
    kid_scores.append(mmd)
  return float(np.mean(kid_scores)), float(np.std(kid_scores))


def calculate_inception_score(logits, splits=10):
  pyx = torch.softmax(logits, dim=1).numpy()
  n = len(pyx)
  split_size = n // splits
  scores = []
  for i in range(splits):
    part = pyx[i * split_size:(i + 1) * split_size]
    py = np.mean(part, axis=0)
    kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
    scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
  return float(np.mean(scores)), float(np.std(scores))


def calculate_coverage_density(real_features, gen_features, k=5):
  D_real = cdist(real_features, real_features)
  np.fill_diagonal(D_real, np.inf)
  radii = np.sort(D_real, axis=1)[:, k - 1]
  D = cdist(real_features, gen_features)
  coverage = float(np.mean(np.min(D, axis=1) <= radii))
  density = float(np.mean(np.sum(D <= radii[:, np.newaxis], axis=1)) / k)
  return coverage, density


class ImageDS(Dataset):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.image_files = [str(f.relative_to(root_dir)) for f in Path(root_dir).rglob('*') if f.suffix.lower() in ('.png', '.jpg')]

    self.transform = transforms.Compose([
        transforms.Resize(342, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def __len__(self):
    return len(self.image_files)


  def __getitem__(self, idx):
    img_path = Path(self.root_dir) / self.image_files[idx]
    image = read_image(str(img_path))
    if image.shape[0] == 1:
      return self.__getitem__(idx+1)
    image = self.transform(image)
    return image


def run_eval_track(inception_model, real_dir, gen_dir, track_name):
  """Run all metrics for one evaluation track. Returns a dict of results."""
  ds_real = ImageDS(real_dir)
  ds_gen = ImageDS(gen_dir)
  dl_real = DataLoader(ds_real, batch_size=32, shuffle=True)
  dl_gen = DataLoader(ds_gen, batch_size=32, shuffle=True)

  features = []
  hook = inception_model.avgpool.register_forward_hook(
      lambda m, inp, out: features.append(out.flatten(1))
  )

  for batch in dl_real:
    with torch.no_grad():
      inception_model(batch)

  gen_logits = []
  for batch in dl_gen:
    with torch.no_grad():
      out = inception_model(batch)
    gen_logits.append(out)

  hook.remove()
  gen_logits = torch.cat(gen_logits)

  real_features = torch.cat(features[:len(dl_real)]).cpu().numpy()
  gen_features = torch.cat(features[len(dl_real):]).cpu().numpy()

  mu_real = np.mean(real_features, axis=0)
  sigma_real = np.cov(real_features, rowvar=False)
  mu_gen = np.mean(gen_features, axis=0)
  sigma_gen = np.cov(gen_features, rowvar=False)

  fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
  kid_mean, kid_std = calculate_kid(real_features, gen_features)
  is_mean, is_std = calculate_inception_score(gen_logits)
  coverage, density = calculate_coverage_density(real_features, gen_features)

  print(f"\n[{track_name}] Real: {len(ds_real)}, Generated: {len(ds_gen)}")
  print(f"  FID:      {fid_score:.4f}")
  print(f"  KID:      {kid_mean:.4f} ± {kid_std:.4f}")
  print(f"  IS:       {is_mean:.4f} ± {is_std:.4f}")
  print(f"  Coverage: {coverage:.4f}, Density: {density:.4f}")

  return {
      f"{track_name}/fid": fid_score,
      f"{track_name}/kid_mean": kid_mean,
      f"{track_name}/kid_std": kid_std,
      f"{track_name}/is_mean": is_mean,
      f"{track_name}/is_std": is_std,
      f"{track_name}/coverage": coverage,
      f"{track_name}/density": density,
      f"{track_name}/num_real_images": len(ds_real),
      f"{track_name}/num_gen_images": len(ds_gen),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, help="Model directory name, e.g. black-forest-labs--FLUX.1-dev")
  parser.add_argument("--base-dir", default="generated_images", help="Base directory containing model subdirectories")
  parser.add_argument("--imagenet-real-dir", default="imagenet_samples", help="Real ImageNet reference images")
  parser.add_argument("--coco-real-dir", default="coco_samples", help="Real COCO reference images")
  args = parser.parse_args()

  model_name = args.model.replace("--", "/")
  model_dir = Path(args.base_dir) / args.model
  imagenet_gen_dir = model_dir / "imagenet"
  coco_gen_dir = model_dir / "coco"

  wandb.init(
      project="fid-eval",
      name=model_name,
      config={
          "model": model_name,
          "feature_layer": "avgpool",
          "feature_dim": 2048,
          "batch_size": 32,
      }
  )

  inception = inception_v3(weights=Inception_V3_Weights)
  inception.eval()

  log_data = {}

  if imagenet_gen_dir.exists():
    log_data.update(run_eval_track(inception, args.imagenet_real_dir, str(imagenet_gen_dir), "imagenet"))
  else:
    print(f"Skipping ImageNet track — {imagenet_gen_dir} not found")

  if coco_gen_dir.exists() and Path(args.coco_real_dir).exists():
    log_data.update(run_eval_track(inception, args.coco_real_dir, str(coco_gen_dir), "coco"))
  elif coco_gen_dir.exists():
    print(f"Skipping COCO track — {args.coco_real_dir} not found (run get_coco_samples.py first)")
  else:
    print(f"Skipping COCO track — {coco_gen_dir} not found")

  wandb.log(log_data)
  wandb.finish()
