import os, sys
sys.path.insert(0, r"D:\GeoROI-YOLO\LabelImgApp")
from labelImg import get_main_app
image_dir = r"D:\待补标验证集"
class_file = r"D:\GeoROI-YOLO\LabelImgApp\data\predefined_classes.txt"
save_dir = r"D:\A-learn\weichat\xwechat_files\wxid_qqa6ttn73nlc22_158f\msg\file\2026-07\CPLID_YOLO\CPLID_YOLO\insulator\labels\val"
os.makedirs(save_dir, exist_ok=True)
app, win = get_main_app(["labelImg.py", image_dir, class_file, save_dir])
sys.exit(app.exec_())
