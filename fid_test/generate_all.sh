#!/bin/bash
# Generate ImageNet and COCO images for all models.
# Run from the fid_test/ directory: bash generate_all.sh
# Safe to rerun — skips any track that already has enough images.
# did this one manually -    "stabilityai/stable-diffusion-xl-base-1.0"

NUM_IMAGES=2500

MODELS=(
    "runwayml/stable-diffusion-v1-5"
    "black-forest-labs/FLUX.1-dev"
    "black-forest-labs/FLUX.1-schnell"
    "stabilityai/stable-diffusion-3.5-large"
    "stabilityai/sdxl-turbo"
)

count_images() {
    find "$1" -name "*.png" 2>/dev/null | wc -l | tr -d ' '
}

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Model: $MODEL"
    echo "========================================"
    MODEL_SLUG="${MODEL//\/\/--}"
    MODEL_SLUG="${MODEL/\///}"
    MODEL_SLUG="${MODEL//\//-}"
    MODEL_SLUG="${MODEL//\//--}"

    IMAGENET_DIR="generated_images/${MODEL_SLUG}/imagenet"
    COCO_DIR="generated_images/${MODEL_SLUG}/coco"

    echo "--- ImageNet track ---"
    if [ "$(count_images "$IMAGENET_DIR")" -ge "$NUM_IMAGES" ]; then
        echo "  Already complete ($NUM_IMAGES images), skipping."
    else
        modal run make_images_imagenet.py --model "$MODEL" || echo "  WARNING: ImageNet track failed for $MODEL"
    fi

    echo "--- COCO track ---"
    if [ "$(count_images "$COCO_DIR")" -ge "$NUM_IMAGES" ]; then
        echo "  Already complete ($NUM_IMAGES images), skipping."
    else
        modal run make_images_coco.py --model "$MODEL" || echo "  WARNING: COCO track failed for $MODEL"
    fi

    echo "Done with $MODEL"
    echo ""
done

echo "All models complete."
