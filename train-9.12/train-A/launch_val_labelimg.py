import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELIMG_DIR = ROOT.parent.parent / "LabelImgApp"
if not LABELIMG_DIR.exists():
    sys.exit(f"找不到 LabelImgApp 目录: {LABELIMG_DIR}\n请确认 LabelImgApp/ 在项目根目录下（与 train-9.12、data-9.12 同级）。")
sys.path.insert(0, str(LABELIMG_DIR))

from labelImg import get_main_app

# TODO: 改成你要补标的验证集图片目录
image_dir = r"D:\待补标验证集"
class_file = LABELIMG_DIR / "data" / "predefined_classes.txt"
# TODO: 改成你希望标签保存到的目录
save_dir = ROOT.parent.parent / "data-9.12" / "obb_dataset" / "labels" / "val"
os.makedirs(save_dir, exist_ok=True)
app, win = get_main_app(["labelImg.py", image_dir, str(class_file), str(save_dir)])
sys.exit(app.exec_())
