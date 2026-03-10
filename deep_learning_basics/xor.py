import torch

neural_network = torch.nn.Sequential(
  torch.nn.Linear(2, 64),
  torch.nn.SiLU(),
  torch.nn.Linear(64, 64),
  torch.nn.SiLU(),
  torch.nn.Linear(64, 1),
)

inputs = torch.randn(1,2) # N, 2
out = neural_network(inputs)

X = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
Y = torch.Tensor([[0],[1],[1],[0]])

def loss_fn(out, y):
  return (out-y).square().mean()

opt = torch.optim.Adam(neural_network.parameters(), lr=1e-3)

for _ in range(10):
  idx = torch.randint(len(X), (1,))
  sample = X[idx].unsqueeze(0)
  target = Y[idx].unsqueeze(0)
  out = neural_network(sample)
  loss = loss_fn(out, target)

  opt.zero_grad()
  loss.backward()
  opt.step()


with torch.no_grad():
  for i in range(len(X)):
    sample = X[i].unsqueeze(0)
    out = neural_network(sample)
    print(f"Input: {X[i].tolist()}, Predicted: {out.item():.4f}, Target: {Y[i].item():.4f}")
