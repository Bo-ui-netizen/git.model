"""训练旋转框（OBB）模型。

标注文件每行必须是：class x1 y1 x2 y2 x3 y3 x4 y4，坐标归一化到 0~1。
"""
from pathlib import Path
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Train an Ultralytics OBB model")
    p.add_argument("--model", default="yolo11n-obb.pt")
    p.add_argument("--data", default="data/cplid_defect_obb.yaml")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0", help="GPU 编号；RTX 4060 通常为 0")
    p.add_argument("--name", default="stage2_defect_obb")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="固定随机种子，便于消融实验公平复现")
    p.add_argument("--patience", type=int, default=100,
                   help="早停耐心轮数；设为 100 可尽量完整训练 100 轮")
    p.add_argument("--resume", default=None, help="从 last.pt 恢复训练")
    a = p.parse_args()
    model = YOLO(a.resume or a.model)
    model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
                device=a.device, project=str(Path(__file__).resolve().parent / "runs"),
                name=a.name, patience=a.patience, workers=a.workers, seed=a.seed,
                resume=bool(a.resume))


if __name__ == "__main__":
    main()
