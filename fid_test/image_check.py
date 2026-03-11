import os
from pathlib import Path
from collections import defaultdict

root = Path("/your/image/dir")
stats = defaultdict(int)

for f in root.rglob("*.jpg"):  # or .png
    # assuming folder structure reflects model name
    stats[f.parent.name] += 1

for model, count in stats.items():
    print(f"{model}: {count} images")
