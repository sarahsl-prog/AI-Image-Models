# AI-Image-Models

A collection of experiments, training scripts, and evaluation infrastructure for diffusion-based image generation models. The repository explores DreamBooth-style personalization with FLUX, flow matching on MNIST, and a systematic multi-model evaluation study using FID, Inception Score, and CLIP Score.

## Repository Layout

| Path | Description |
| --- | --- |
| `dreambooth-app.py` | Modal-deployed DreamBooth LoRA fine-tuning of `black-forest-labs/FLUX.1-dev`, packaged with a Gradio web UI for inference. |
| `train.py` | Top-level training script. |
| `instance_example_urls.txt` | URLs to the personalization images consumed by `dreambooth-app.py`. |
| `assets/` | Source images used for personalization and web UI assets (icon, CSS, background). |
| `deep_learning_basics/` | Minimal NN primers — XOR example and a small training loop with a short glossary. |
| `experiment_tracking/` | Flow matching on MNIST with Weights & Biases sweep configs (`sweep_lr.yaml`, `sweep_formulation.yaml`) and a small CNN evaluator. |
| `fid_test/` | Standalone FID computation harness with multiple implementations (`fid.py`, `fid_pytorch.py`, `fid_tensor.py`) and ImageNet/COCO sample generators. |
| `Testing_project/` | Multi-model image generation evaluation study — FID/IS/CLIP across FLUX.1-dev, FLUX.1-schnell, SD v1.5, and SDXL, plus a model-collapse experiment loop. See `Testing_project/README.md` for the full write-up. |
| `pyproject.toml` / `uv.lock` | Root project dependencies, managed with [uv](https://docs.astral.sh/uv/). |

## Getting Started

The root project targets Python 3.11+ and is managed with `uv`:

```bash
uv sync
```

The DreamBooth app runs on [Modal](https://modal.com). After authenticating with `modal token new` and setting up `huggingface-secret` (and optionally `wandb-secret`), train and serve with:

```bash
modal run dreambooth-app.py            # fine-tune
modal serve dreambooth-app.py          # launch the Gradio UI
```

Subprojects (`Testing_project/`, `fid_test/`, `experiment_tracking/`) each have their own dependency files and READMEs — see those directories for project-specific instructions.

## License

Released under the [MIT License](LICENSE).
