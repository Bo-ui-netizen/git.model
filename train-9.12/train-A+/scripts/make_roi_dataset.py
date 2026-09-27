from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x

from common import ensure_dir, enlarge_box, load_yolo_labels, save_yolo_labels, yolo_to_xyxy, xyxy_to_yolo


def intersect_area(a, b) -> float:
    """计算两个 xyxy 框的相交面积；完全不相交时返回 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def convert_defects_to_crop(labels, crop_box, image_w, image_h, min_keep=0.25):
    """把原图缺陷标签变换到 ROI 内；保留至少 25% 落在裁剪区的缺陷。"""
    cx1, cy1, cx2, cy2 = crop_box
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    out = []
    for label in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(label, image_w, image_h)
        if cls != 1:  # 原始 full 数据集中：0 是绝缘子，1 是缺陷。
            continue
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if intersect_area((x1, y1, x2, y2), crop_box) / area < min_keep:
            continue
        nx1 = max(0.0, x1 - cx1)
        ny1 = max(0.0, y1 - cy1)
        nx2 = min(float(crop_w), x2 - cx1)
        ny2 = min(float(crop_h), y2 - cy1)
        if nx2 > nx1 and ny2 > ny1:
            # ROI 数据集只有“缺陷”一种类别，因此类别号重新映射为 0。
            out.append((0, *xyxy_to_yolo((nx1, ny1, nx2, ny2), crop_w, crop_h)))
    return out


def process_split(src_root: Path, out_root: Path, split: str, padding: float, size: int):
    """处理 train 或 val：每张全图按绝缘子框裁成一个或多个固定尺寸 ROI。"""
    img_dir = src_root / "images" / split
    label_dir = src_root / "labels" / split
    out_img_dir = ensure_dir(out_root / "images" / split)
    out_label_dir = ensure_dir(out_root / "labels" / split)

    for image_path in tqdm(sorted(img_dir.iterdir()), desc=f"roi:{split}"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = Image.open(image_path).convert("RGB")
        image_w, image_h = image.size
        labels = load_yolo_labels(label_dir / f"{image_path.stem}.txt")
        # 从 full 标签中筛出绝缘子框；[1:] 丢掉类别号，只保留四个坐标。
        insulators = [yolo_to_xyxy(row, image_w, image_h)[1:] for row in labels if row[0] == 0]
        if not insulators:
            insulators = [(0, 0, image_w, image_h)]
        for idx, ins_box in enumerate(insulators):
            crop_box = enlarge_box(ins_box, image_w, image_h, padding)
            x1, y1, x2, y2 = crop_box
            if x2 <= x1 or y2 <= y1:
                continue
            # 放大后的小缺陷在像素上更显著，适合第二阶段模型学习。
            resized = image.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
            roi_labels = convert_defects_to_crop(labels, crop_box, image_w, image_h)
            roi_name = f"{image_path.stem}_roi{idx:02d}.jpg"
            resized.save(out_img_dir / roi_name, quality=95)
            scaled_labels = []
            for cls, xc, yc, bw, bh in roi_labels:
                scaled_labels.append((cls, xc, yc, bw, bh))
            save_yolo_labels(out_label_dir / f"{Path(roi_name).stem}.txt", scaled_labels)


def main():
    # 此脚本离线生成阶段二训练数据，不参与模型推理。
    parser = argparse.ArgumentParser(description="Create defect ROI dataset from full CPLID YOLO labels.")
    parser.add_argument("--src", default="datasets/CPLID_YOLO/full")
    parser.add_argument("--out", default="datasets/CPLID_ROI")
    parser.add_argument("--padding", type=float, default=0.30)
    parser.add_argument("--size", type=int, default=768)
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    for split in ("train", "val"):
        process_split(src_root, out_root, split, args.padding, args.size)
    print(f"ROI dataset saved to {out_root}")


if __name__ == "__main__":
    main()
