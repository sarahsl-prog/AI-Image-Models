# Flow Matching on MNIST — Experiment Tracking

- **`test.py`** — `make_small_cnn()` returns a Sequential, `make_eval_entropy()` loads weights and returns a closure you just call with images. Main block trains to 99% and saves `mnist_cnn.pt`.
- **`train.py`** — argparse for `--lr`, `--epochs`, `--formulation {jit,rectified}`. Logs loss, entropy, and sample grid to wandb every epoch. Rectified flow just predicts `v = x - e` directly — no `(1-t)` denominator or clipping.
- **`sweep_lr.yaml`** — grid over lr `[1e-3, 3e-4, 1e-4]` with JiT formulation
- **`sweep_formulation.yaml`** — JiT vs rectified at lr `3e-4`

## Usage

```
python test.py                          # train eval CNN first
wandb sweep sweep_lr.yaml               # then launch either sweep
wandb sweep sweep_formulation.yaml
wandb agent <sweep-id>
```
