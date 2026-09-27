from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

# 处理数据集时认可的图像扩展名。
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在；parents=True 表示父目录不存在时一并创建。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_images(root: Union[str, Path]) -> Iterable[Path]:
    """递归遍历目录，逐个产出图片路径（yield 类似按需返回）。"""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMG_EXTS:
            yield path


def read_image(path: Union[str, Path]):
    """用 OpenCV 读取图片，读取失败则立即报出清晰错误。"""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def xyxy_to_yolo(box, width: int, height: int):
    """将像素框 (x1,y1,x2,y2) 转成 YOLO 标签 (中心点x,中心点y,宽,高)。"""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    # YOLO 标签使用 0~1 的相对坐标，而不是图片中的像素坐标。
    return xc / width, yc / height, bw / width, bh / height


def yolo_to_xyxy(label, width: int, height: int):
    """将 YOLO 标签反算为像素框，便于裁剪图片和画框。"""
    cls, xc, yc, bw, bh = label
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return int(cls), x1, y1, x2, y2


def clip_box(box, width: int, height: int):
    """将框限制在图像范围内，防止数组裁剪越界。"""
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def enlarge_box(box, width: int, height: int, padding: float):
    """向四周等比例扩展框；padding=0.30 表示宽高总共额外增加约 30%。"""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * padding / 2.0
    dy = bh * padding / 2.0
    return clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def load_yolo_labels(path: Union[str, Path]):
    """读取 YOLO txt 标注；每行格式为 class xc yc width height。"""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        rows.append((int(float(parts[0])), *map(float, parts[1:])))
    return rows


def save_yolo_labels(path: Union[str, Path], rows):
    """以 YOLO 规定的六位小数格式写入标签文件。"""
    lines = []
    for row in rows:
        cls, xc, yc, bw, bh = row
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
