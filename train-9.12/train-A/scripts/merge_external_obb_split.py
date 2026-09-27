"""Merge an external manually labeled OBB split, treating missing JSON as empty labels."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def rows_from_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = data["imageWidth"], data["imageHeight"]
    rows = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if shape.get("label") != "defect" or len(points) != 4:
            raise ValueError(f"Invalid shape in {path.name}")
        values = []
        for x, y in points:
            values += [min(1, max(0, x / width)), min(1, max(0, y / height))]
        rows.append("0 " + " ".join(f"{v:.6f}" for v in values))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-images", required=True, type=Path)
    p.add_argument("--source-json", required=True, type=Path)
    p.add_argument("--dest-images", required=True, type=Path)
    p.add_argument("--dest-labels", required=True, type=Path)
    args = p.parse_args()
    args.dest_images.mkdir(parents=True, exist_ok=True)
    args.dest_labels.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in args.source_images.iterdir() if path.is_file())
    conflicts = [p.name for p in images if (args.dest_images / p.name).exists()]
    if conflicts:
        raise FileExistsError(f"Filename conflicts: {conflicts[:5]}")
    objects = empty = 0
    for image in images:
        shutil.copy2(image, args.dest_images / image.name)
        source_json = args.source_json / f"{image.stem}.json"
        target_label = args.dest_labels / f"{image.stem}.txt"
        if source_json.exists():
            rows = rows_from_json(source_json)
            target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            objects += len(rows)
        else:
            target_label.write_text("", encoding="utf-8")
            empty += 1
    print(f"Merged images={len(images)}, OBB objects={objects}, empty-label images={empty}")


if __name__ == "__main__":
    main()
