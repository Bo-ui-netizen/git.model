"""Convert four-point LabelMe polygons to Ultralytics YOLO-OBB labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--class-name", default="defect")
    args = parser.parse_args()
    args.label_dir.mkdir(parents=True, exist_ok=True)
    converted = skipped = 0
    for json_path in sorted(args.json_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        width, height = data["imageWidth"], data["imageHeight"]
        rows = []
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if shape.get("label") != args.class_name or len(points) != 4:
                skipped += 1
                continue
            coords = []
            for x, y in points:
                coords += [min(1, max(0, x / width)), min(1, max(0, y / height))]
            rows.append("0 " + " ".join(f"{value:.6f}" for value in coords))
        (args.label_dir / f"{json_path.stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        converted += 1
    print(f"Converted {converted} JSON files; skipped {skipped} non-four-point shapes.")


if __name__ == "__main__":
    main()
