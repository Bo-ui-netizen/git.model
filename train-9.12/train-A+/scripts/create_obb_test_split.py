"""从当前验证集按文件名稳定拆出独立测试集，避免 test 与 val 使用同一张图。"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ROOT = Path("obb_dataset")


def is_test(name: str) -> bool:
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % 5 == 0


def main():
    val_images, val_labels = ROOT / "images" / "val", ROOT / "labels" / "val"
    test_images, test_labels = ROOT / "images" / "test", ROOT / "labels" / "test"
    test_images.mkdir(parents=True, exist_ok=True)
    test_labels.mkdir(parents=True, exist_ok=True)
    moved = 0
    for image in val_images.glob("*"):
        if image.is_file() and is_test(image.name):
            shutil.move(str(image), str(test_images / image.name))
            label = val_labels / f"{image.stem}.txt"
            if label.exists():
                shutil.move(str(label), str(test_labels / label.name))
            moved += 1
    print(f"Moved {moved} images from val to independent test")


if __name__ == "__main__":
    main()
