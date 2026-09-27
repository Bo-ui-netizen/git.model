"""Convert Labelme insulator polygons to binary entity masks."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--json-dir',default='obb_dataset/images/train'); p.add_argument('--out',default='datasets/CPLID_ROI/masks/train'); p.add_argument('--manifest',default='mask_review/train/manifest.csv'); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    stems=None
    if Path(a.manifest).exists():
        stems={Path(r['image']).stem for r in csv.DictReader(Path(a.manifest).open(encoding='utf-8-sig'))}
    made=skipped=0
    for jp in sorted(Path(a.json_dir).glob('*.json')):
        if stems is not None and jp.stem not in stems: continue
        try: d=json.loads(jp.read_text(encoding='utf-8'))
        except Exception: skipped+=1; continue
        h,w=int(d.get('imageHeight',0)),int(d.get('imageWidth',0))
        if not h or not w: skipped+=1; continue
        mask=np.zeros((h,w),np.uint8)
        for s in d.get('shapes',[]):
            if s.get('label')!='insulator' or s.get('shape_type')!='polygon': continue
            pts=np.asarray(s.get('points',[]),np.float32)
            if len(pts)>=3: cv2.fillPoly(mask,[np.round(pts).astype(np.int32)],255)
        if not np.any(mask): skipped+=1; continue
        cv2.imwrite(str(out/f'{jp.stem}.png'),mask); made+=1
    print(f'Precise masks written: {made}; skipped: {skipped}; output: {out}')
if __name__=='__main__': main()
