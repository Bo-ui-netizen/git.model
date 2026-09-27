"""批量生成 OBB 第二阶段 JPG 预览图。"""
from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


def draw_obb(image, corners, label):
    points = corners.astype("int32").reshape((-1, 1, 2))
    cv2.polylines(image, [points], True, (0, 0, 255), 2)
    x, y = points[0, 0]
    cv2.putText(image, label, (int(x), max(20, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="obb_dataset/images/val")
    p.add_argument("--labels", default="obb_dataset/labels/val")
    p.add_argument("--model", default="runs/stage2_defect_obb_finetune100/weights/best.pt")
    p.add_argument("--out", default="stage_preview/batch10_final150")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=768)
    a = p.parse_args()
    image_dir, label_dir, output_dir = Path(a.images), Path(a.labels), Path(a.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [f for f in sorted(label_dir.glob("*.txt")) if f.stat().st_size > 0][:a.count]
    model = YOLO(a.model)
    for index, label_path in enumerate(selected, 1):
        image_path = image_dir / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path))
        result = model.predict(image, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        count = 0
        if result.obb is not None:
            for poly, score, cls in zip(result.obb.xyxyxyxy.cpu().numpy(), result.obb.conf.cpu().numpy(), result.obb.cls.cpu().numpy()):
                draw_obb(image, poly, f"defect {int(cls)} {float(score):.2f}")
                count += 1
        output_path = output_dir / f"{index:02d}_{image_path.stem}_stage2.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"{index:02d}: {output_path} detections={count}")


if __name__ == "__main__":
    main()
