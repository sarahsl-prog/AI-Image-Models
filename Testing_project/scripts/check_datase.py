from datasets import load_dataset
ds = load_dataset("phiyodr/coco2017", split="validation", streaming=True)
for item in ds:
    print(dict({k: v for k, v in item.items() if k != 'image'}))
    break

