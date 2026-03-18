# Build Project: Multi-Model Image Generation Evaluation Study

## Overview

This project is a systematic evaluation of four state-of-the-art image generation models using a pre-built dataset of 14,000 generated images paired with real reference images from two benchmark datasets. Rather than training a new model from scratch, the focus is on the **evaluation and measurement infrastructure** that forms the real day-to-day work of ML engineers — writing tests, tracking experiments, and iterating on findings.

---

## Models Under Evaluation

|Model|Type|Notes|
|---|---|---|
|`black-forest-labs/FLUX.1-dev`|Flow Matching|High-quality, slower inference|
|`black-forest-labs/FLUX.1-schnell`|Flow Matching (distilled)|Fast inference, quality tradeoff|
|`runwayml/stable-diffusion-v1-5`|Latent Diffusion|Established baseline|
|`stabilityai/stable-diffusion-xl-base-1.0`|Latent Diffusion (XL)|Larger architecture, higher resolution|

---

## Dataset

### Generated Images

- **14,000 total images** across 4 models
- Each model generated:
    - **2,500 images** from COCO captions (text-to-image)
    - **1,000 images** from ImageNet class labels (class-conditional)
- Images are **index-matched**: `0000.png` across all models corresponds to the same prompt, enabling paired comparisons

### Reference Images

- **2,500 real COCO val2017 images** — used as the ground truth reference set for COCO-track evaluation
- **1,000 real ImageNet samples** — used as the ground truth reference set for ImageNet-track evaluation
    - Class labels are encoded in filenames (e.g., `00000_label856.png`) enabling per-class analysis

### Prompt Recovery

COCO captions are fully recoverable via the `phiyodr/coco2017` HuggingFace dataset. The original generation script streamed the validation split sequentially with no random seed, so the 2,500 prompts are deterministic and reproducible. A recovery script will export `coco_prompts.json` mapping each index to its original caption and image ID.

ImageNet class names are directly derivable from the label IDs in filenames using the standard ImageNet class index.

---

## Evaluation Metrics

### 1. Fréchet Inception Distance (FID)

- **What it measures:** Distribution-level similarity between generated and real images using InceptionV3 feature statistics
- **Lower is better**
- **Computed for:** Each model × each dataset (COCO and ImageNet) = 8 FID scores
- **Additional cross-model FID:** Comparing generated distributions against each other to measure inter-model similarity (relevant to the model collapse research question)

### 2. Inception Score (IS)

- **What it measures:** Both quality (sharpness of class predictions) and diversity of generated images
- **Higher is better**
- **Limitation:** Biased toward ImageNet-like content; more meaningful for the ImageNet track

### 3. CLIP Score (COCO track only)

- **What it measures:** Semantic alignment between generated image and source caption using a CLIP model
- **Higher is better**
- **Requires:** Recovered COCO prompts (see above)
- **Computed for:** Each model's 2,500 COCO-generated images vs. their source captions

### 4. Per-Class FID / Quality Analysis (ImageNet track)

- Subset FID computed per ImageNet class to identify which classes each model handles well or poorly
- Enables comparison like: _"FLUX.1-dev outperforms SDXL on fine-grained animal classes but underperforms on abstract concepts"_

---

## Infrastructure

### Experiment Tracking

All runs will be tracked using **Weights & Biases (W&B)** or **MLflow** (TBD — this comparison may itself be a documented finding in the report). Tracked artifacts will include:

- Per-model metric scores
- Sample image grids per model
- Experiment configuration and runtime
- Comparison plots
** local MLflow - export MLFLOW_TRACKING_URI=http://localhost:5000
**
### Compute Strategy

|Phase|Where|Rationale|
|---|---|---|
|Prompt recovery, setup|Local (RTX 2000 Ada 16GB)|Lightweight, no GPU needed|
|FID/IS computation|Local|InceptionV3 inference is fast locally|
|CLIP scoring|Local|Fits in 16GB VRAM comfortably|
|Any retraining experiments|Modal (A10G) → then local|Burn Modal credits on heavy runs first|

Code will be written to be **portable between local and Modal** from the start. This is a first-class engineering concern.

### Directory Structure (planned)

```
project/
├── data/
│   ├── generated_images/          # 4 model dirs × coco + imagenet
│   ├── coco_samples/              # 2500 real reference images
│   ├── imagenet_samples/          # 1000 real reference images
│   └── coco_prompts.json          # recovered captions
├── evaluation/
│   ├── fid.py                     # FID computation
│   ├── inception_score.py         # IS computation
│   ├── clip_score.py              # CLIP scoring
│   └── per_class.py               # ImageNet class-level analysis
├── tracking/
│   └── experiment_logger.py       # W&B / MLflow abstraction layer
├── tests/
│   └── test_output_quality.py     # Formal test(s) per project requirements
├── scripts/
│   ├── get_coco_samples.py        # Reference image downloader (existing)
│   ├── recover_coco_prompts.py    # Caption recovery script
│   └── run_eval.py                # Main evaluation entrypoint
└── report/
    └── findings.md
```

---

## Experiment Loop

The project requirements call for at least one test and one experiment. The plan is to run the full loop multiple times:

### Loop 1 — Baseline Evaluation

- Compute FID, IS for all 4 models on both datasets
- Compute CLIP scores for all 4 models on COCO track
- Log everything to experiment tracker
- **Finding:** Which model wins on which metric?

### Loop 2 — Per-Class Deep Dive (ImageNet)

- Identify 5–10 ImageNet classes with highest variance across models
- Visualize generated samples side-by-side
- Compute class-level FID for those classes
- **Finding:** Do models have systematic strengths/weaknesses by category?

### Loop 3 — Cross-Model Distribution Analysis

- Compute pairwise FID between model outputs (not just vs. real images)
- Quantify how similar FLUX.1-dev vs FLUX.1-schnell are to each other
- **Finding:** Does the schnell distillation meaningfully alter the output distribution?

### Loop 4 (Optional / if time allows) — Synthetic Training Experiment

- Train a small DDPM on one model's generated images
- Compare outputs and FID against a model trained on real images
- Directly tests the **model collapse** hypothesis
- Can be run on Modal for heavy training, local for evaluation

---

## Formal Test Requirement

Per the project spec, at least one formal test evaluating model output is required. Planned:

```python
# test_output_quality.py
def test_fid_below_threshold():
    """
    Assert that at least one model achieves FID < [threshold]
    on the COCO validation set, confirming the evaluation
    pipeline produces meaningful results.
    """
    ...

def test_clip_score_above_chance():
    """
    Assert that CLIP scores for generated images are
    statistically higher than random image-caption pairings.
    """
    ...
```

---

## Report Outline

1. **Introduction** — motivation, dataset description, models evaluated
2. **Evaluation Infrastructure** — metrics chosen, experiment tracking setup, compute strategy
3. **Results: COCO Track** — FID, CLIP scores, visual examples
4. **Results: ImageNet Track** — FID, IS, per-class breakdown
5. **Cross-Model Analysis** — pairwise distribution comparisons
6. **Experiment Iterations** — what was tested, what changed, what improved
7. **Conclusions** — which model wins and under what conditions, infrastructure lessons learned
8. **Appendix** — full metric tables, sample grids

---

## Open Questions

- **Experiment tracker:** W&B vs MLflow — may document this choice with a brief comparison in the report
- **FID sample size:** Using all 2500/1000 images or a fixed subset for fair comparison?
- **CLIP model:** `openai/clip-vit-base-patch32` vs `openai/clip-vit-large-patch14` — larger is better but slower
- **Loop 4 scope:** Model collapse experiment is ambitious; only pursue if loops 1–3 complete with time to spare