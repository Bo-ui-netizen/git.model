from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x

from common import ensure_dir, load_yolo_labels, read_image, save_yolo_labels, xyxy_to_yolo


def irregular_polygon(cx, cy, rw, rh, points=9):
    """生成不规则多边形，作为合成缺陷的大致轮廓。"""
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    random.shuffle(angles)
    poly = []
    for angle in sorted(angles):
        radius = random.uniform(0.55, 1.0)
        x = cx + np.cos(angle) * rw * radius
        y = cy + np.sin(angle) * rh * radius
        poly.append([int(x), int(y)])
    return np.array(poly, dtype=np.int32)


def synthesize(image, entity_mask=None, templates=None, rng=None, position_jitter=0.03, size_jitter=0.12, appearance="dark", strength=(0.45, 0.70), max_attempts=30):
    """随机在无缺陷 ROI 上制造模糊、暗斑或裂纹式缺陷，并返回其外接框。"""
    h, w = image.shape[:2]
    rng = rng or random
    if entity_mask is not None and np.any(entity_mask):
        ys, xs = np.where(entity_mask > 0)
        ex1, ey1, ex2, ey2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ew, eh = ex2 - ex1, ey2 - ey1
    else:
        ex1, ey1, ex2, ey2, ew, eh = 0, 0, w, h, w, h
    for _ in range(max_attempts):
        if templates and entity_mask is not None:
            tx, ty, tw, th = rng.choice(templates)
            scale = rng.uniform(max(0.01, 1 - size_jitter), 1 + size_jitter)
            defect_w, defect_h = tw * ew * scale, th * eh * scale
            cx = ex1 + (tx + rng.uniform(-position_jitter, position_jitter)) * ew
            cy = ey1 + (ty + rng.uniform(-position_jitter, position_jitter)) * eh
        else:
            defect_w = rng.uniform(0.05, 0.14) * ew
            defect_h = rng.uniform(0.05, 0.18) * eh
            cx = rng.uniform(0.25, 0.75) * w
            cy = rng.uniform(0.20, 0.80) * h
        poly = irregular_polygon(cx, cy, defect_w / 2, defect_h / 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        if entity_mask is not None:
            mask = cv2.bitwise_and(mask, (entity_mask > 0).astype(np.uint8) * 255)
        if entity_mask is None or cv2.countNonZero(mask) >= 0.8 * max(1, cv2.contourArea(poly.astype(np.float32))):
            break
    else:
        return None
    mode = appearance
    out = image.copy()
    if mode == "erase":
        blur = cv2.GaussianBlur(image, (31, 31), 0)
        out[mask > 0] = blur[mask > 0]
    elif mode == "dark":
        shadow = out.astype(np.float32)
        shadow[mask > 0] *= rng.uniform(*strength)
        out = np.clip(shadow, 0, 255).astype(np.uint8)
    else:
        pts = poly.reshape(-1, 2)
        for i in range(len(pts) - 1):
            cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), (20, 20, 20), random.randint(1, 3), cv2.LINE_AA)
        out = cv2.GaussianBlur(out, (3, 3), 0)

    x, y, bw, bh = cv2.boundingRect(mask)
    return out, (x, y, x + bw, y + bh)


def main():
    # 只从空标签 ROI 中选图合成，避免覆盖原本真实的缺陷标注。
    parser = argparse.ArgumentParser(description="Generate physically inspired synthetic defect ROIs.")
    parser.add_argument("--src", default="datasets/CPLID_ROI")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ratio", type=float, default=0.5, help="Fraction of negative ROI images to synthesize.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["ordinary", "constrained"], default="ordinary")
    parser.add_argument("--mask-dir", default=None, help="Entity masks (same-stem PNG), required in constrained mode.")
    parser.add_argument("--stats-split", default="train")
    parser.add_argument("--position-jitter", type=float, default=0.03)
    parser.add_argument("--size-jitter", type=float, default=0.12)
    parser.add_argument("--appearance", choices=["erase", "dark", "crack"], default="dark")
    parser.add_argument("--strength-min", type=float, default=0.45)
    parser.add_argument("--strength-max", type=float, default=0.70)
    args = parser.parse_args()

    random.seed(args.seed)  # 固定随机种子，使同一参数下的结果可复现。
    np.random.seed(args.seed)
    root = Path(args.src)
    img_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    ensure_dir(img_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else root / "masks" / args.split
    templates = None
    if args.mode == "constrained":
        templates = []
        stats_img = root / "images" / args.stats_split
        stats_lbl = root / "labels" / args.stats_split
        stats_mask = root / "masks" / args.stats_split
        for p in sorted(stats_img.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            im = read_image(p); mh, mw = im.shape[:2]
            em = cv2.imread(str(stats_mask / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if em is None: continue
            ys, xs = np.where(em > 0)
            if not len(xs): continue
            ex1, ey1, ex2, ey2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
            for _, xc, yc, bw, bh in load_yolo_labels(stats_lbl / f"{p.stem}.txt"):
                templates.append(((xc*mw-ex1)/(ex2-ex1), (yc*mh-ey1)/(ey2-ey1), bw*mw/(ex2-ex1), bh*mh/(ey2-ey1)))
        if not templates:
            raise RuntimeError("constrained mode requires real labels and masks in --stats-split")
    ensure_dir(label_dir)

    candidates = []
    for image_path in sorted(img_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        labels = load_yolo_labels(label_dir / f"{image_path.stem}.txt")
        if not labels:  # 空标签表示这张 ROI 没有真实缺陷，是可用负样本。
            candidates.append(image_path)
    random.shuffle(candidates)
    take = int(len(candidates) * args.ratio)

    for image_path in tqdm(candidates[:take], desc="synth"):
        image = read_image(image_path)
        entity = None
        if args.mode == "constrained":
            entity = cv2.imread(str(mask_dir / f"{image_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if entity is None:
                continue
        result = synthesize(image, entity, templates, random, args.position_jitter, args.size_jitter, args.appearance, (args.strength_min, args.strength_max))
        if result is None:
            continue
        out, box = result
        h, w = out.shape[:2]
        out_name = f"{image_path.stem}_synth.jpg"
        cv2.imwrite(str(img_dir / out_name), out)
        # ROI 阶段只有一类：类别 0 代表 defect。
        save_yolo_labels(label_dir / f"{Path(out_name).stem}.txt", [(0, *xyxy_to_yolo(box, w, h))])
    print(f"Synthetic defects added: {take} ({args.mode})")


if __name__ == "__main__":
    main()
