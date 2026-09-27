from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    # 验证不会更新模型参数，只会在验证集上计算 Precision、Recall、mAP 等指标。
    parser = argparse.ArgumentParser(description="Validate a YOLO model and print metrics.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="GPU 编号，RTX 4060 通常为 0")
    args = parser.parse_args()

    model = YOLO(args.model)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(metrics)


if __name__ == "__main__":
    main()
