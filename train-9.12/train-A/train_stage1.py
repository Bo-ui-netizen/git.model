from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    # 阶段一只学习“绝缘子”这一类，为后续 ROI 裁剪提供位置。
    parser = argparse.ArgumentParser(description="Train stage-1 insulator detector.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default="data/cplid_insulator.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="GPU 编号，RTX 4060 通常为 0")
    parser.add_argument("--name", default="stage1_insulator")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(args.model)  # 加载预训练 YOLO；不是从零开始训练。
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).resolve().parent / "runs"),
        name=args.name,
        workers=args.workers,
        patience=20,  # 验证集连续 20 轮没有变好时提前停止。
    )


if __name__ == "__main__":
    main()
