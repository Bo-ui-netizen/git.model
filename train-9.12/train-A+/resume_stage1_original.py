from ultralytics import YOLO

model = YOLO('runs/stage1_original_gpu100/weights/last.pt')
model.train(resume=True, device='0', workers=0)
