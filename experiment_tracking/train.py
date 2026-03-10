"""flow matching on mnist — JiT and rectified flow, with wandb tracking"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from einops import rearrange
import wandb

from test import make_eval_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def time_embed(t, dim=128):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(torch.log(torch.tensor(10000.0)) / half))
    args = t[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


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


@torch.no_grad()
def sample(model, formulation, n=64, steps=300):
    z = torch.randn(n, 784, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_cur = torch.full((n,), i * dt, device=device).clamp(1e-5, 1 - 1e-5)
        out = model(z, t_cur)
        if formulation == "jit":
            v = (out - z) / (1 - t_cur[:, None])
        else:  # rectified flow: model directly predicts v
            v = out
        z = z + dt * v
    return rearrange(z.clamp(0, 1), "b (h w) -> b 1 h w", h=28, w=28)


def train(args):
    wandb.init(project="flow-mnist", config=vars(args))
    cfg = wandb.config

    train_ds = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = FlowMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    eval_entropy = make_eval_entropy()

    for epoch in range(cfg.epochs):
        total_loss = 0
        for x, _ in train_dl:
            x = rearrange(x, "b 1 h w -> b (h w)").to(device)
            e = torch.randn_like(x)
            t = torch.sigmoid(torch.randn(x.shape[0], device=device))
            z = t[:, None] * x + (1 - t[:, None]) * e

            if cfg.formulation == "jit":
                one_minus_t = (1 - t[:, None]).clamp(min=0.05)
                v_target = (x - z) / one_minus_t
                x_pred = model(z, t)
                v_pred = (x_pred - z) / one_minus_t
                loss = (v_target - v_pred).pow(2).mean()
            else:  # rectified flow
                v_target = x - e
                v_pred = model(z, t)
                loss = (v_target - v_pred).pow(2).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        images = sample(model, cfg.formulation, n=64)
        entropy = eval_entropy(images)

        wandb.log({"loss": avg_loss, "entropy": entropy, "samples": wandb.Image(make_grid(images, nrow=8))})
        print(f"epoch {epoch:02d} | loss {avg_loss:.4f} | entropy {entropy:.4f}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--formulation", choices=["jit", "rectified"], default="jit")
    args = parser.parse_args()
    train(args)
