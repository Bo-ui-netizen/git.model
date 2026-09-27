"""Prepare an offline review set for manually correcting OBB-derived masks."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', default='obb_dataset/images/train')
    p.add_argument('--masks', default='datasets/CPLID_ROI/masks/train')
    p.add_argument('--out', default='mask_review/train')
    p.add_argument('--limit', type=int, default=168)
    a = p.parse_args(); out=Path(a.out); overlay_dir=out/'overlays'; overlay_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; count=0
    for ip in sorted(Path(a.images).iterdir()):
        if ip.suffix.lower() not in EXTS or count >= a.limit: continue
        mp=Path(a.masks)/f'{ip.stem}.png'
        im=cv2.imread(str(ip)); mask=cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if im is None or mask is None: continue
        if mask.shape[:2] != im.shape[:2]:
            mask=cv2.resize(mask,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mp),mask)
        overlay=im.copy(); red=np.zeros_like(im); red[:,:,2]=255
        alpha=(mask>0)[:,:,None].astype(np.float32)*0.35
        overlay=(overlay*(1-alpha)+red*alpha).astype(np.uint8)
        cv2.imwrite(str(overlay_dir/f'{ip.stem}_overlay.jpg'),overlay)
        rows.append({'image':str(ip),'mask':str(mp),'overlay':str(overlay_dir/f'{ip.stem}_overlay.jpg'),'status':'TODO'})
        count+=1
    with (out/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['image','mask','overlay','status']); w.writeheader(); w.writerows(rows)
    (out/'README.txt').write_text('用 Labelme 打开 image，沿绝缘子真实外轮廓修正 mask。白色=绝缘子，黑色=背景；保存为同名 PNG。完成后将 manifest.csv 的 status 改为 DONE。overlay 用于快速检查。\n',encoding='utf-8')
    print(f'Review package: {count} samples -> {out}')
if __name__=='__main__': main()
