#!/bin/bash
# Generate ImageNet and COCO images for all models.
# Run from the fid_test/ directory: bash generate_all.sh

set -e

MODELS=(
    "stabilityai/stable-diffusion-xl-base-1.0"
    "runwayml/stable-diffusion-v1-5"
    "black-forest-labs/FLUX.1-dev"
    "black-forest-labs/FLUX.1-schnell"
    "stabilityai/stable-diffusion-3.5-large"
    "stabilityai/sdxl-turbo"
)

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Model: $MODEL"
    echo "========================================"

    echo "--- ImageNet track ---"
    modal run make_images_imagenet.py --model "$MODEL"

    echo "--- COCO track ---"
    modal run make_images_coco.py --model "$MODEL"

    echo "Done with $MODEL"
    echo ""
done

echo "All models complete."
