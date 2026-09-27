from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
model = YOLO(str(ROOT / "runs" / "stage1_insulator_gpu100" / "weights" / "best.pt"))
# TODO: 改成你自己的验证图路径
img = r"D:\A-learn\weichat\xwechat_files\wxid_qqa6ttn73nlc22_158f\msg\file\2026-07\CPLID_YOLO\CPLID_YOLO\insulator\images\val\0765.jpg"
out = ROOT / "runs" / "stage1_insulator_gpu100" / "preview_0765"
model.predict(source=img, device=0, imgsz=640, conf=0.25, save=True,
              project=str(out), name='result', exist_ok=True)
print(Path(out, 'result').resolve())
