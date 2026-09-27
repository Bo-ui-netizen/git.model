"""Select up to 500 external InsuFault-Net images containing defect classes.

Classes in the source release: 0=insulator, 1=damaged, 2=Flashover, 3=hammer.
Only classes 1 and 2 are selected because they are defect categories.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFECT_CLASSES = {"1": "damaged", "2": "Flashover"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", default=500, type=int)
    args = parser.parse_args()
    image_root, label_root = args.source / "images", args.source / "labels"
    selected = []
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(image_root)
        label_path = label_root / relative.with_suffix(".txt")
        if not label_path.exists():
            continue
        labels = [line.split()[0] for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        defect_types = sorted({DEFECT_CLASSES[label] for label in labels if label in DEFECT_CLASSES})
        if defect_types:
            selected.append((image_path, label_path, ",".join(defect_types)))
        if len(selected) >= args.count:
            break
    if len(selected) < args.count:
        raise RuntimeError(f"Only found {len(selected)} defect images, expected {args.count}.")
    image_out, label_out = args.output / "images", args.output / "source_labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source_split", "filename", "defect_types", "source_label"])
        for index, (image_path, label_path, defect_types) in enumerate(selected, 1):
            name = f"{index:03d}_{image_path.name}"
            shutil.copy2(image_path, image_out / name)
            shutil.copy2(label_path, label_out / f"{Path(name).stem}.txt")
            writer.writerow([index, image_path.parent.name, name, defect_types, label_path])
    print(f"Selected {len(selected)} external defect images into {args.output}")


if __name__ == "__main__":
    main()
