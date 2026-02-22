import random
from pathlib import Path

import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fid import compute_fid


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


if __name__ == "__main__":
  generated_dir = 'generated_images/black-forest-labs--FLUX.1-dev'
  # generated_dir = 'generated_images/runwayml--stable-diffusion-v1-5'

  wandb.init(
      project="fid-eval",
      config={
          "real_dir": "imagenet_samples",
          "generated_dir": generated_dir,
          "feature_layer": "avgpool",
          "feature_dim": 2048,
          "batch_size": 32,
      }
  )

  ds = ImageDS('imagenet_samples')
  sd15 = ImageDS(generated_dir)

  dl = DataLoader(ds, batch_size=32, shuffle=True)
  dl_sd15 = DataLoader(sd15, batch_size=32, shuffle=True)

  model = inception_v3(weights=Inception_V3_Weights)
  model.eval()

  features = []
  hook = model.avgpool.register_forward_hook(
      lambda m, inp, out: features.append(out.flatten(1))
  )

  for batch in dl:
    print(batch.shape)
    with torch.no_grad():
      model(batch)
    out_real = features.pop()
    break

  for batch in dl_sd15:
    with torch.no_grad():
      model(batch)
    out_gen = features.pop()
    break

  hook.remove()

  fid_score = compute_fid(out_gen, out_real)
  print(fid_score)
  wandb.log({"fid": fid_score})
  wandb.finish()
