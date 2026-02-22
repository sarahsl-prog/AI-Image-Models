import torch


def matrix_sqrt(M):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    return eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T


def matrix_sqrt_product(sigma_g, sigma_r):
    """Compute sqrtm(sigma_g @ sigma_r) via symmetric decomposition."""
    sqrt_r = matrix_sqrt(sigma_r)
    M = sqrt_r @ sigma_g @ sqrt_r
    return matrix_sqrt(M)


def frechet_distance(mu_g, sigma_g, mu_r, sigma_r):
  a = (mu_g - mu_r).square().sum()
  b = torch.trace(sigma_g + sigma_r - 2*matrix_sqrt_product(sigma_g, sigma_r))
  return a+b


def feature_stats(feats): # (N, 2048)
  mu = feats.mean(dim=0)
  sigma = torch.cov(feats.T)
  return mu, sigma


def compute_fid(gen_features, real_features):
  mu_g, sigma_g = feature_stats(gen_features)
  mu_r, sigma_r = feature_stats(real_features)
  return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def compute_fid_gt(gen_features, real_features, fe=torch.nn.Identity()):
  from ignite.metrics import FID
  from ignite.engine import Engine
  def eval_step(engine, batch):
      return batch
  default_evaluator = Engine(eval_step)
  metric = FID(num_features=2048, feature_extractor=fe)
  metric.attach(default_evaluator, "fid")
  state = default_evaluator.run([[gen_features, real_features]])
  return state.metrics["fid"]


if __name__ == "__main__":
  dummy_features_real = torch.randn(32, 2048) # nn output
  dummy_features_gen = torch.randn(32, 2048)

  fid_gt = compute_fid_gt(dummy_features_gen, dummy_features_real)
  fid_dev = float(fid(dummy_features_gen, dummy_features_real))
  print(fid_gt)
  print(fid_dev)
  # assert abs(fid_gt-fid_dev) < 1., "both metrics are not equivalent"
