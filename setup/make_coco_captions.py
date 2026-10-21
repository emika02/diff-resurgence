"""Make the --neutral-data CSV for loss_stability.py from COCO val2017.

    python setup/make_coco_captions.py data/coco data/coco_neutral.csv --n 512

Picks n random images (fixed seed), one caption each. Image paths are written relative to the CSV.
"""
import argparse
import csv
import json
import os
import random

p = argparse.ArgumentParser()
p.add_argument("coco_dir", help="folder containing val2017/ and annotations/")
p.add_argument("out_csv")
p.add_argument("--n", type=int, default=512)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

with open(os.path.join(args.coco_dir, "annotations", "captions_val2017.json")) as f:
    ann = json.load(f)
files = {im["id"]: im["file_name"] for im in ann["images"]}
caps = {}
for a in ann["annotations"]:
    caps.setdefault(a["image_id"], []).append(" ".join(a["caption"].split()))

ids = sorted(caps)
random.Random(args.seed).shuffle(ids)
out_dir = os.path.dirname(os.path.abspath(args.out_csv))
rows = []
for i in ids[:args.n]:
    path = os.path.join(os.path.abspath(args.coco_dir), "val2017", files[i])
    if os.path.exists(path):
        rows.append((os.path.relpath(path, out_dir), sorted(caps[i])[0]))

with open(args.out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "caption"])
    w.writerows(rows)
print(f"wrote {len(rows)} image/caption pairs to {args.out_csv}")