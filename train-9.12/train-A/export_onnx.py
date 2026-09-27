from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    # ONNX 是跨框架部署格式，便于后续用 C++、ONNX Runtime 或 TensorRT 推理。
    parser = argparse.ArgumentParser(description="Export trained YOLO model to ONNX.")
    parser.add_argument("--model", default="runs/detect/stage2_defect_roi/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    # dynamic=True 时导出的模型可接受不同尺寸的输入，但部署端处理会更复杂。
    model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=args.dynamic)


if __name__ == "__main__":
    main()
