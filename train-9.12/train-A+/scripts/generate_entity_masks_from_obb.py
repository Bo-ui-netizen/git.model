"""Rasterize OBB polygons into approximate insulator entity masks."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', default='obb_dataset/images/train')
    p.add_argument('--labels', default='obb_dataset/labels/train')
    p.add_argument('--out', default='datasets/CPLID_ROI/masks/train')
    p.add_argument('--class-id', type=int, default=0)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    for ip in sorted(Path(a.images).iterdir()):
        if ip.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}: continue
        im = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if im is None: skipped += 1; continue
        h,w = im.shape; mask = np.zeros((h,w), np.uint8)
        lp = Path(a.labels) / f'{ip.stem}.txt'
        if lp.exists():
            for line in lp.read_text(encoding='utf-8').splitlines():
                q=line.split()
                if len(q)!=9 or int(float(q[0]))!=a.class_id: continue
                pts=np.asarray([float(v) for v in q[1:]], np.float32).reshape(4,2)
                pts[:,0]*=w; pts[:,1]*=h
                cv2.fillPoly(mask,[np.round(pts).astype(np.int32)],255)
        if not np.any(mask): skipped += 1; continue
        cv2.imwrite(str(out/f'{ip.stem}.png'),mask); made += 1
    print(f'Entity masks written: {made}; skipped: {skipped}; output: {out}')
if __name__ == '__main__': main()
