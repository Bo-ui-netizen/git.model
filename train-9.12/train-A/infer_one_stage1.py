from ultralytics import YOLO
from pathlib import Path
model=YOLO(r"D:\GeoROI-YOLO\runs\stage1_insulator_gpu100\weights\best.pt")
img=r"D:\A-learn\weichat\xwechat_files\wxid_qqa6ttn73nlc22_158f\msg\file\2026-07\CPLID_YOLO\CPLID_YOLO\insulator\images\val\0765.jpg"
out=r"D:\GeoROI-YOLO\runs\stage1_insulator_gpu100\preview_0765"
model.predict(source=img, device=0, imgsz=640, conf=0.25, save=True, project=out, name='result', exist_ok=True)
print(Path(out,'result').resolve())
