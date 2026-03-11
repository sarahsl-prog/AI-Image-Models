from datasets import load_dataset
import json
import random

ds = load_dataset("phiyodr/coco2017", split="validation", streaming=True)

captions = []
for item in ds:
    raw = item.get("captions", [])
    if raw:
        captions.append({
            "image_id": item.get("image_id"),
            "caption": raw[0],
        })
    if len(captions) >= 5000:
        break

random.seed(42)
random.shuffle(captions)
captions = captions[:2500]

prompts = [{"index": i, **c} for i, c in enumerate(captions)]

with open("coco_prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"Recovered {len(prompts)} prompts")
