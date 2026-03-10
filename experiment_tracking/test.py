"""Tiny CNN trained to 99%+ on MNIST. Provides eval_entropy() for scoring generated images."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def make_small_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def make_eval_entropy(weights_path="mnist_cnn.pt"):
    model = make_small_cnn().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    @torch.no_grad()
    def eval_entropy(images):
        logits = model(images.to(device))
        return torch.distributions.Categorical(logits=logits).entropy().mean().item()

    return eval_entropy


if __name__ == "__main__":
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    model = make_small_cnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"epoch {epoch} | acc {acc:.4f}")
        if acc >= 0.99:
            break

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print(f"saved mnist_cnn.pt (acc={acc:.4f})")
