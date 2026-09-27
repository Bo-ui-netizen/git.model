# 两阶段推理入口：先在原图找绝缘子，再在每个绝缘子 ROI 内找缺陷。
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from scripts.common import enlarge_box, ensure_dir, read_image
from scripts.geo_postprocess import fuse_predictions


'''def result_boxes(result):
    """将 Ultralytics 的结果对象转换为统一的 (x1,y1,x2,y2,置信度,类别) 元组列表。"""
    boxes = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        boxes.append((x1, y1, x2, y2, conf, cls))
    return boxes'''
def draw_obb(image, corners, color=(0, 0, 255)):
    pts = corners.astype("int32").reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

result = model.predict(image)[0]

if result.obb is not None:
    polygons = result.obb.xyxyxyxy.cpu().numpy()  # N×4×2，像素坐标
    for corners in polygons:
        draw_obb(vis, corners)

def draw_box(image, box, color, label):
    """在 OpenCV 图片上画检测框和文字；OpenCV 的颜色顺序是 BGR。"""
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    # argparse 负责读取命令中的 --image、--stage1 等参数。
    parser = argparse.ArgumentParser(description="Two-stage Geo-ROI-YOLO inference pipeline.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--stage1", default="runs/detect/stage1_insulator/weights/best.pt")
    parser.add_argument("--stage2", default="runs/detect/stage2_defect_roi/weights/best.pt")
    parser.add_argument("--out", default="outputs/infer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--padding", type=float, default=0.30)
    parser.add_argument("--roi-size", type=int, default=768)
    parser.add_argument("--conf1", type=float, default=0.25)
    parser.add_argument("--conf2", type=float, default=0.20)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = read_image(image_path)
    h, w = image.shape[:2]  # 图像数组的形状是 (高, 宽, 通道数)。
    vis = image.copy()
    out_dir = ensure_dir(args.out)

    stage1 = YOLO(args.stage1)  # 阶段一：全图检测绝缘子。
    stage2 = YOLO(args.stage2)  # 阶段二：放大后的 ROI 内检测缺陷。
    ins_result = stage1.predict(image, imgsz=640, conf=args.conf1, device=args.device, verbose=False)[0]
    ins_boxes = result_boxes(ins_result)

    all_defects = []
    for idx, ins in enumerate(ins_boxes):
        # 每个绝缘子分别裁剪为 ROI；idx 用于给保存的 ROI 文件编号。
        crop_box = enlarge_box(ins[:4], w, h, args.padding)
        x1, y1, x2, y2 = crop_box
        # NumPy 图像切片按 [y(行), x(列)] 书写，和坐标 (x, y) 的顺序不同。
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # 统一 ROI 尺寸能放大微小缺陷，也让阶段二模型收到固定尺度的输入。
        roi = cv2.resize(crop, (args.roi_size, args.roi_size), interpolation=cv2.INTER_LINEAR)
        defect_result = stage2.predict(roi, imgsz=args.roi_size, conf=args.conf2, device=args.device, verbose=False)[0]
        roi_preds = result_boxes(defect_result)
        roi_preds = fuse_predictions(roi, roi_preds)
        # 阶段二框的坐标属于 768x768 ROI；sx/sy 用于映射回原始图片。
        sx = (x2 - x1) / args.roi_size
        sy = (y2 - y1) / args.roi_size
        for dx1, dy1, dx2, dy2, score, cls in roi_preds:
            # 先撤销 ROI 缩放（乘 sx/sy），再加上裁剪框左上角的原图偏移。
            full_box = (x1 + dx1 * sx, y1 + dy1 * sy, x1 + dx2 * sx, y1 + dy2 * sy, score, cls)
            all_defects.append(full_box)
        draw_box(vis, (x1, y1, x2, y2), (0, 180, 255), f"insulator {ins[4]:.2f}")
        cv2.imwrite(str(out_dir / f"{image_path.stem}_roi{idx:02d}.jpg"), roi)

    for defect in all_defects:
        draw_box(vis, defect, (0, 0, 255), f"defect {defect[4]:.2f}")

    out_path = out_dir / f"{image_path.stem}_geo_roi_result.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved {out_path}; insulators={len(ins_boxes)}, defects={len(all_defects)}")


# 直接执行此文件时启动 main；被其他文件 import 时不自动推理。
if __name__ == "__main__":
    main()
