'''
Per-Class FID / Quality Analysis (ImageNet track)
Subset FID computed per ImageNet class to identify which classes each model handles well or poorly
Enables comparison like: "FLUX.1-dev outperforms SDXL on fine-grained animal classes but underperforms on abstract concepts"
Requires: Class labels for each generated image (from COCO prompts or ImageNet metadata)
Computed for: Each model × each ImageNet class = up to 1000 FID scores (can be averaged or visualized as a distribution)

Directory layout expected:
    real_dir/
        Affenpinscher/   ← one subdir per class
        Afghan Hound/
        ...
    gen_dir/
        Affenpinscher/
        Afghan Hound/
        ...
'''

from pathlib import Path

from evaluation.fid import evaluate_fid


# ── private helpers ───────────────────────────────────────────────────────────

def _get_class_dirs(image_dir):
    '''Returns {class_name: Path} for each subdirectory.'''
    return {d.name: d for d in sorted(Path(image_dir).iterdir()) if d.is_dir()}


# ── public API ────────────────────────────────────────────────────────────────

def evaluate_per_class(real_dir, gen_dir, logger=None, min_images=10):
    '''
    Compute FID for each ImageNet class subdirectory shared between real_dir and gen_dir.
    Skips classes with fewer than min_images in either split (FID is unreliable on tiny sets).
    Returns dict of {class_name: fid_score} (lower is better).
    Logs each class score to logger if provided.
    '''
    real_classes = _get_class_dirs(real_dir)
    gen_classes  = _get_class_dirs(gen_dir)
    shared = sorted(real_classes.keys() & gen_classes.keys())

    scores = {}
    for cls in shared:
        real_imgs = list(real_classes[cls].glob('*'))
        gen_imgs  = list(gen_classes[cls].glob('*'))
        if len(real_imgs) < min_images or len(gen_imgs) < min_images:
            continue
        log_key = f'fid_{cls.replace(" ", "_")}'
        scores[cls] = evaluate_fid(real_classes[cls], gen_classes[cls],
                                   logger=logger, log_key=log_key)

    return scores
