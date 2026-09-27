from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    # baseline 是对照实验：直接在整张图上同时检测绝缘子和缺陷。
    parser = argparse.ArgumentParser(description="Train full-image YOLO baseline for comparison.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default="data/cplid_full.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="GPU 编号，RTX 4060 通常为 0")
    parser.add_argument("--name", default="baseline_full_yolo11n")
    args = parser.parse_args()

    model = YOLO(args.model)  # 读取预训练权重（默认 yolo11n.pt）。
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).resolve().parent / "runs"),  # 训练结果保存位置。
        name=args.name,
        patience=20,
    )


if __name__ == "__main__":
    main()
