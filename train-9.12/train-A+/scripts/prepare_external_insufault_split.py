"""Create a reproducible manual-OBB labeling split from selected InsuFault images."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-count", default=150, type=int)
    parser.add_argument("--val-count", default=350, type=int)
    parser.add_argument("--seed", default=20260820, type=int)
    args = parser.parse_args()
    manifest = args.source / "manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    if args.train_count + args.val_count > len(rows):
        raise ValueError("Requested more images than available")
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["defect_types"]].append(row)
    rng = random.Random(args.seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    train, val = [], []
    total = len(rows)
    for bucket in buckets.values():
        train_size = round(len(bucket) * args.train_count / total)
        train.extend(bucket[:train_size])
        val.extend(bucket[train_size:])
    rng.shuffle(train)
    rng.shuffle(val)
    while len(train) < args.train_count:
        train.append(val.pop())
    while len(train) > args.train_count:
        val.append(train.pop())
    val = val[:args.val_count]
    for split, split_rows in (("train", train), ("val", val)):
        image_out = args.output / "images" / split
        source_label_out = args.output / "source_labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        source_label_out.mkdir(parents=True, exist_ok=True)
        with (args.output / f"{split}_manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=split_rows[0].keys())
            writer.writeheader()
            for row in split_rows:
                filename = row["filename"]
                shutil.copy2(args.source / "images" / filename, image_out / filename)
                shutil.copy2(args.source / "source_labels" / f"{Path(filename).stem}.txt", source_label_out / f"{Path(filename).stem}.txt")
                writer.writerow(row)
        print(f"{split}: {len(split_rows)} images")


if __name__ == "__main__":
    main()
