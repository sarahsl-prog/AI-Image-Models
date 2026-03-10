"""flow matching on mnist with an mlp"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# data: mnist normalized to [0, 1]
train_ds = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True) #, drop_last=True)

# sinusoidal time embedding
def time_embed(t, dim=128):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(torch.log(torch.tensor(10000.0)) / half))
    args = t[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)

# mlp that takes (z, t_emb) and predicts x directly (JiT)
class FlowMLP(nn.Module):
    def __init__(self, dim=784, t_dim=128, h=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + t_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, dim),
        )
    def forward(self, z, t):
        return self.net(torch.cat([z, time_embed(t)], dim=-1))

model = FlowMLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

# train: z = t*x + (1-t)*e, network predicts x, loss on derived velocity
for epoch in range(100):
    total_loss = 0
    for x, _ in train_dl:
        x = rearrange(x, "b 1 h w -> b (h w)").to(device)
        e = torch.randn_like(x)
        # logit-normal t sampling (paper Appendix A, mu=0 sigma=1 for MNIST)
        t = torch.sigmoid(torch.randn(x.shape[0], device=device))
        z = t[:, None] * x + (1 - t[:, None]) * e
        # paper clips (1-t) denominator to 0.05 to avoid zero division
        one_minus_t = (1 - t[:, None]).clamp(min=0.05)
        v = (x - z) / one_minus_t
        x_pred = model(z, t)
        v_pred = (x_pred - z) / one_minus_t
        loss = (v - v_pred).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    print(f"epoch {epoch:02d} | loss {total_loss / len(train_dl):.4f}")

# sample: euler integrate from noise (t=0) to data (t=1)
@torch.no_grad()
def sample(n=25, steps=300):
    z = torch.randn(n, 784, device=device)
    frames = []
    for i in range(steps):
      # TODO homework
    return frames

frames = sample()
Path("samples").mkdir(exist_ok=True)
from torchvision.utils import make_grid
from PIL import Image

pil_frames = []
for frame in frames:
    grid = make_grid(frame, nrow=5)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    pil_frames.append(Image.fromarray(grid_np))

pil_frames[0].save(
    "samples/flow_mnist.gif",
    save_all=True,
    append_images=pil_frames[1:],
    duration=20,
    loop=0,
)
pil_frames[-1].save("samples/flow_mnist_final.png")
print("saved samples/flow_mnist.gif")
