#!/usr/bin/env bash
# run_all_evals.sh
# Runs all evaluation scripts for all models and generates the full report.
# Must be run from the Testing_project root directory:
#   bash scripts/run_all_evals.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python}"

log() { echo -e "\n\033[1;34m==> $*\033[0m"; }
die() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

cd "$PROJECT_ROOT"

MODELS=(flux-dev flux-schnell sd15 sdxl)

# ── Step 1: per-model evaluation (FID, CLIP, IS) ──────────────────────────────
for model in "${MODELS[@]}"; do
    log "run_eval.py — $model"
    "$PYTHON" scripts/run_eval.py --model "$model" --dataset all \
        || die "run_eval.py failed for $model"
done

# ── Step 2: cross-model comparison charts ─────────────────────────────────────
log "plot_results.py"
"$PYTHON" scripts/plot_results.py \
    || die "plot_results.py failed"

# ── Step 3: per-class variance analysis ───────────────────────────────────────
log "class_variance_analysis.py"
"$PYTHON" scripts/class_variance_analysis.py \
    || die "class_variance_analysis.py failed"

# ── Step 4: pairwise cross-model FID ──────────────────────────────────────────
log "cross_model_fid.py"
"$PYTHON" scripts/cross_model_fid.py \
    || die "cross_model_fid.py failed"

log "All done. Reports saved to report/"
