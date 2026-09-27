"""把普通 YOLO 水平框复制为 OBB 的四点格式。

这是兼容转换：源标签没有旋转角度，生成的四点框也必然是水平的。
要训练真正的旋转框，必须在标注软件中将四个点调整到目标实际边缘。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_line(line: str) -> str:
    values = line.split()
    if len(values) != 5:
        raise ValueError(f"Expected 5 values, got {len(values)}: {line!r}")
    cls, cx, cy, width, height = values
    cx, cy, width, height = map(float, (cx, cy, width, height))
    x1, y1 = cx - width / 2, cy - height / 2
    x2, y2 = cx + width / 2, cy + height / 2
    # 顺时针四点，坐标限制在合法的归一化范围内。
    points = (x1, y1, x2, y1, x2, y2, x1, y2)
    points = [min(1.0, max(0.0, p)) for p in points]
    return cls + " " + " ".join(f"{p:.6f}" for p in points)


def copy_split(source_root: Path, output_root: Path, split: str) -> tuple[int, int]:
    source_images = source_root / "images" / split
    source_labels = source_root / "labels" / split
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    image_count = label_count = 0
    for image in source_images.iterdir():
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(image, output_images / image.name)
            image_count += 1
            label = source_labels / f"{image.stem}.txt"
            target = output_labels / label.name
            if not label.exists():
                target.write_text("", encoding="utf-8")
                continue
            rows = [convert_line(line) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            label_count += len(rows)
    return image_count, label_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", default="data/cplid_defect_obb", type=Path)
    args = p.parse_args()
    total_images = total_labels = 0
    for split in ("train", "val"):
        images, labels = copy_split(args.source, args.output, split)
        total_images += images
        total_labels += labels
        print(f"{split}: {images} images, {labels} labels")
    print(f"Created {args.output.resolve()}: {total_images} images, {total_labels} labels")


if __name__ == "__main__":
    main()
