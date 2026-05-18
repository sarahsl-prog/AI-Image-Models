# AI-Image-Models

A workspace for the **Build Fellowship** project *"AI Image Models: Get Started with the Basics"* — an 8-week, intermediate-level project led by Armand du Parc Locmaria (Data Science Fellow, Open Avenues Foundation) that focuses on the day-to-day infrastructure work of an ML engineer: writing tests for model outputs, tracking experiments, and iterating quickly on image generation models.

> Project page: [AI Image Models: Get Started with the Basics](https://hub.buildfellowship.com/projects/ai-image-models-get-started-with-the-basics)

## Project Goal

In the coming years, generative AI will be extremely useful for robotics. Training robots in the real world is prohibitively expensive, so we need simulators — and classical simulators don't match real-world distributions. The aim of this project is to build a simple learned simulator in the form of an image generation model, with the engineering rigor that real production ML work demands.

By the end of the project you will have:

- Trained and fine-tuned an image generation model on a subject of your choice.
- Written at least one **test** evaluating model outputs (e.g. an FID distribution metric).
- Run at least one **experiment** to try to improve the model, tracked with Weights & Biases.
- Written a short **report** with your findings.

## Workshop Schedule (8 weeks)

| # | Workshop | Focus |
| --- | --- | --- |
| 1 | Introduction | Learned simulators, the ML engineer day-to-day, deliverables, available compute. |
| 2 | Train Your First Model! | Fine-tune an image generation model on a subject of your choice. |
| 3 | Is the Model Good? Writing our First Test | Implement the FID metric to evaluate image quality. |
| 4 | Neural Network Basics | Neural networks, gradient descent, batching, parameters, hyperparameters. |
| 5 | Running and Tracking Experiments | Tweak hyperparameters; thoroughly track experiments in Weights & Biases. |
| 6 | Hands on! | Clone the template repository and set up your own experiments. |
| 7 | Iteration Speed Over Everything | Write a Model Flops Utilization (MFU) test to measure hardware efficiency. |
| 8 | Presentations | Present your experiments and conclusions. |

## Prerequisites

- Programming experience (e.g. a Python script that scrapes a website or a small game).
- Basic debugging skills (Python debugger, reading docs/datasheets, prompting LLMs).
- General CS fundamentals: `git`, `bash`, terminal use.
- Interest in AI, simulators, and ML.

## Repository Layout

This repo collects the work for the project — the template starting point, the per-week explorations, and the final multi-model evaluation study.

| Path | Description |
| --- | --- |
| `dreambooth-app.py` | Modal-deployed DreamBooth LoRA fine-tuning of `black-forest-labs/FLUX.1-dev` with a Gradio web UI for inference. The "train your first model" entry point. |
| `train.py` | Top-level training script. |
| `instance_example_urls.txt` | URLs to the personalization images consumed by `dreambooth-app.py`. |
| `assets/` | Source images for personalization plus web UI assets (icon, CSS, background). |
| `deep_learning_basics/` | Minimal NN primers — XOR example, a small training loop, and a short glossary. |
| `experiment_tracking/` | Flow matching on MNIST with W&B sweep configs (`sweep_lr.yaml`, `sweep_formulation.yaml`) and a small CNN evaluator. |
| `fid_test/` | Standalone FID computation harness with multiple implementations (`fid.py`, `fid_pytorch.py`, `fid_tensor.py`) and ImageNet/COCO sample generators. |
| `Testing_project/` | The multi-model evaluation study — FID/IS/CLIP across FLUX.1-dev, FLUX.1-schnell, SD v1.5, and SDXL, plus a model-collapse experiment loop. See `Testing_project/README.md` for the full write-up. |
| `pyproject.toml` / `uv.lock` | Root project dependencies, managed with [uv](https://docs.astral.sh/uv/). |

## Getting Started

The root project targets Python 3.11+ and is managed with `uv`:

```bash
uv sync
```

The DreamBooth app runs on [Modal](https://modal.com). After authenticating with `modal token new` and configuring `huggingface-secret` (and optionally `wandb-secret`), train and serve with:

```bash
modal run dreambooth-app.py            # fine-tune
modal serve dreambooth-app.py          # launch the Gradio UI
```

Subprojects (`Testing_project/`, `fid_test/`, `experiment_tracking/`) each ship their own dependency files and READMEs — check those directories for project-specific instructions.

## About the Fellow

**Armand du Parc Locmaria** — Data Science Build Fellow at Open Avenues. Armand is a machine learning engineer at comma.ai working on deep learning infrastructure (making model training fast and easy). He previously made deep learning educational videos at Weights & Biases and quantified dog aging at LoyalForDogs. He holds a Master's degree in Computer Science.

## License

Released under the [MIT License](LICENSE).
