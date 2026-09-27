"""自动检测绝缘子并生成 YOLO 水平框标签。

单张图片示例：
  python auto_label_insulator.py --image input.jpg
文件夹示例：
  python auto_label_insulator.py --image D:/images --out D:/auto_labels

输出：
  out/labels/*.txt       YOLO 格式：class xc yc w h（归一化）
  out/preview/*.jpg      带检测框的可视化图片
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def draw_box(image, xyxy, conf):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 255), 2)
    cv2.putText(image, f"insulator {conf:.2f}", (x1, max(22, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)


def label_one(model, image_path: Path, labels_dir: Path, preview_dir: Path,
              conf_threshold: float, device: str, imgsz: int) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[跳过] 无法读取: {image_path}")
        return 0
    height, width = image.shape[:2]
    preview = image.copy()
    result = model.predict(image, imgsz=imgsz, conf=conf_threshold,
                           device=device, verbose=False)[0]

    rows = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu().numpy())
            # 像素坐标转换为 YOLO 归一化中心点格式。
            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            draw_box(preview, (x1, y1, x2, y2), score)

    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{image_path.stem}.txt"
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    preview_path = preview_dir / f"{image_path.stem}.jpg"
    cv2.imwrite(str(preview_path), preview)
    print(f"{image_path.name}: 检测到 {len(rows)} 个绝缘子 -> {label_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Auto-label insulators with a YOLO detector")
    parser.add_argument("--image", required=True, help="单张图片或图片文件夹")
    parser.add_argument("--model", default="runs/stage1_original_gpu100/weights/best.pt")
    parser.add_argument("--out", default="outputs/auto_insulator_labels")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="检测置信度阈值；小目标/远景可尝试 0.03~0.10")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0", help="GPU 编号，RTX 4060 通常为 0")
    args = parser.parse_args()

    source = Path(args.image)
    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    else:
        raise FileNotFoundError(source)
    if not images:
        raise RuntimeError(f"没有找到图片: {source}")

    model = YOLO(args.model)
    out = Path(args.out)
    total = sum(label_one(model, p, out / "labels", out / "preview",
                          args.conf, args.device, args.imgsz) for p in images)
    print(f"完成：{len(images)} 张图片，共检测到 {total} 个绝缘子")
    print(f"标签目录：{(out / 'labels').resolve()}")
    print(f"预览目录：{(out / 'preview').resolve()}")


if __name__ == "__main__":
    main()
