from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x

from common import ensure_dir, iter_images, xyxy_to_yolo


# 数据集可能用不同词描述同一种目标；统一映射为项目内部类别号。
CLASS_ALIASES = {
    "insulator": 0,
    "insulator_string": 0,
    "normal_insulator": 0,
    "defect": 1,
    "defective": 1,
    "broken": 1,
    "missing": 1,
}


def find_xml(image_path: Path, raw_root: Path) -> Optional[Path]:
    """按几个常见目录规则为图片寻找同名 VOC XML 标注文件。"""
    candidates = [
        image_path.with_suffix(".xml"),
        raw_root / "Annotations" / f"{image_path.stem}.xml",
        raw_root / "annotations" / f"{image_path.stem}.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    hits = list(raw_root.rglob(f"{image_path.stem}.xml"))
    return hits[0] if hits else None


def normalize_class(name: str) -> Optional[int]:
    """将原始类别名称归一化：绝缘子=0，缺陷=1，未知类别返回 None。"""
    key = name.strip().lower().replace(" ", "_")
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if "defect" in key or "broken" in key or "missing" in key:
        return 1
    if "insulator" in key:
        return 0
    return None


def parse_voc(xml_path: Path):
    """读取一份 Pascal VOC XML，返回图片宽高和有效的像素坐标框。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    objects = []
    for obj in root.findall("object"):
        cls = normalize_class(obj.findtext("name", ""))
        box = obj.find("bndbox")
        if cls is None or box is None:
            continue
        x1 = float(box.findtext("xmin", "0"))
        y1 = float(box.findtext("ymin", "0"))
        x2 = float(box.findtext("xmax", "0"))
        y2 = float(box.findtext("ymax", "0"))
        if x2 > x1 and y2 > y1:
            objects.append((cls, x1, y1, x2, y2))
    return width, height, objects


def write_split(items, out_root: Path, split: str, mode: str):
    """将一组图像/XML 写成 YOLO 的 images/labels/train 或 val 目录结构。"""
    img_dir = ensure_dir(out_root / mode / "images" / split)
    label_dir = ensure_dir(out_root / mode / "labels" / split)
    for image_path, xml_path in tqdm(items, desc=f"{mode}:{split}"):
        width, height, objects = parse_voc(xml_path)
        if width <= 0 or height <= 0:
            continue
        dst_img = img_dir / image_path.name
        shutil.copy2(image_path, dst_img)
        labels = []
        for cls, x1, y1, x2, y2 in objects:
            if mode == "insulator" and cls != 0:
                continue
            # insulator 专用数据集只有一类，所以原绝缘子类别 0 仍映射为 0。
            mapped_cls = 0 if mode == "insulator" else cls
            labels.append((mapped_cls, *xyxy_to_yolo((x1, y1, x2, y2), width, height)))
        label_path = label_dir / f"{image_path.stem}.txt"
        label_path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels), encoding="utf-8")


def main():
    # 将原始 CPLID 的 VOC(XML、像素坐标) 格式转换为 YOLO(txt、归一化坐标) 格式。
    parser = argparse.ArgumentParser(description="Convert CPLID VOC XML annotations to YOLO format.")
    parser.add_argument("--raw", default="datasets/InsulatorDataSet", help="Path to cloned CPLID repository.")
    parser.add_argument("--out", default="datasets/CPLID_YOLO", help="Output YOLO dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    pairs = []
    for image_path in iter_images(raw_root):
        xml_path = find_xml(image_path, raw_root)
        if xml_path:
            pairs.append((image_path, xml_path))
    if not pairs:
        raise SystemExit(f"No image/XML pairs found under {raw_root}. Did you clone CPLID first?")

    random.seed(args.seed)  # 固定划分，保证以后可得到同一训练/验证集。
    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_ratio))
    val_items = pairs[:val_count]
    train_items = pairs[val_count:]

    for mode in ("full", "insulator"):
        write_split(train_items, out_root, "train", mode)
        write_split(val_items, out_root, "val", mode)
    print(f"Converted {len(pairs)} images to {out_root}")


if __name__ == "__main__":
    main()
