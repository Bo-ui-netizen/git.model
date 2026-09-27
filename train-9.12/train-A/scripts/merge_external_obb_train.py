"""Safely merge manually labeled external OBB images into a training split."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def json_rows(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = data["imageWidth"], data["imageHeight"]
    rows = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if shape.get("label") != "defect" or len(points) != 4:
            raise ValueError(f"Invalid shape in {path.name}")
        coords = []
        for x, y in points:
            coords += [min(1, max(0, x / width)), min(1, max(0, y / height))]
        rows.append("0 " + " ".join(f"{value:.6f}" for value in coords))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-images", required=True, type=Path)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--dest-images", required=True, type=Path)
    parser.add_argument("--dest-labels", required=True, type=Path)
    args = parser.parse_args()
    args.dest_images.mkdir(parents=True, exist_ok=True)
    args.dest_labels.mkdir(parents=True, exist_ok=True)
    images = [path for path in sorted(args.source_images.iterdir()) if path.is_file()]
    conflicts = [path.name for path in images if (args.dest_images / path.name).exists()]
    if conflicts:
        raise FileExistsError(f"Image filename conflicts: {conflicts[:5]}")
    written = empty = 0
    for image in images:
        shutil.copy2(image, args.dest_images / image.name)
        json_path = args.source_json / f"{image.stem}.json"
        label_path = args.dest_labels / f"{image.stem}.txt"
        if json_path.exists():
            rows = json_rows(json_path)
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            written += len(rows)
        else:
            label_path.write_text("", encoding="utf-8")
            empty += 1
    print(f"Merged images={len(images)}, OBB objects={written}, empty-label images={empty}")


if __name__ == "__main__":
    main()
