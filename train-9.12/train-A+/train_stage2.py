from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    # 阶段二在已裁剪、放大的绝缘子 ROI 中只学习检测缺陷。
    parser = argparse.ArgumentParser(description="Train stage-2 defect detector on ROI crops.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default="data/cplid_defect_roi.yaml")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0", help="GPU 编号，RTX 4060 通常为 0")
    parser.add_argument("--name", default="stage2_defect_roi")
    args = parser.parse_args()

    model = YOLO(args.model)  # 加载预训练 YOLO。
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).resolve().parent / "runs"),
        name=args.name,
        patience=30,
        close_mosaic=10,  # 训练最后 10 轮关闭拼图增强，使训练更接近真实 ROI。
    )


if __name__ == "__main__":
    main()
