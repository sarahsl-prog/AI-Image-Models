import random
from pathlib import Path

import torch
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
  ds = ImageDS('imagenet_samples')
  # sd15 = ImageDS('generated_images/runwayml--stable-diffusion-v1-5')
  sd15 = ImageDS('generated_images/black-forest-labs--FLUX.1-dev')

  dl = DataLoader(ds, batch_size=32, shuffle=True)
  dl_sd15 = DataLoader(sd15, batch_size=32, shuffle=True)

  model = inception_v3(weights=Inception_V3_Weights)

  for batch in dl:
    print(batch.shape)
    out_real = model(batch).logits
    break

  for batch in dl_sd15:
    out_gen = model(batch).logits
    break

  print(compute_fid(out_gen, out_real))
  # 1799 for sd 1.5
  # 1729 for flux
