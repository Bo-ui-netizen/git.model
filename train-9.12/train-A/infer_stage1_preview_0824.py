from ultralytics import YOLO
model = YOLO(r"D:\GeoROI-YOLO\runs\stage1_insulator_gpu100\weights\best.pt")
model.predict(
    source=r"D:\A-learn\weichat\xwechat_files\wxid_qqa6ttn73nlc22_158f\msg\file\2026-07\CPLID_YOLO\CPLID_YOLO\insulator\images\val\0824.jpg",
    device=0, imgsz=640, conf=0.25, save=True,
    project=r"D:\GeoROI-YOLO\runs\stage1_insulator_gpu100\preview_0824", name="result", exist_ok=True)
