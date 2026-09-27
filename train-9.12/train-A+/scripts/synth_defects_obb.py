"""Generate OBB synthetic defects with ordinary or entity-mask-constrained placement."""
from __future__ import annotations
import argparse, random, shutil
from pathlib import Path
import cv2
import numpy as np

EXTS={'.jpg','.jpeg','.png','.bmp'}

def labels(path):
    if not path.exists(): return []
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        p=line.split()
        if len(p)==9: out.append((int(float(p[0])),np.asarray([float(x) for x in p[1:]],np.float32).reshape(4,2)))
    return out

def mask_for(path, shape):
    m=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if m is None:return None
    if m.shape!=shape:m=cv2.resize(m,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return (m>0).astype(np.uint8)

def obb_from_contour(c,w,h):
    rect=cv2.minAreaRect(c.astype(np.float32)); pts=cv2.boxPoints(rect)
    pts[:,0]/=w; pts[:,1]/=h
    return pts.reshape(-1)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--src',default='obb_dataset'); p.add_argument('--out',default='obb_dataset_synth'); p.add_argument('--split',default='train'); p.add_argument('--ratio',type=float,default=.5); p.add_argument('--mode',choices=['ordinary','constrained'],default='constrained'); p.add_argument('--mask-dir',default='datasets/CPLID_ROI/masks/train'); p.add_argument('--restrict-to-masks',action='store_true'); p.add_argument('--size-jitter',type=float,default=.12); p.add_argument('--position-jitter',type=float,default=.03); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); src=Path(a.src); out=Path(a.out)
    for d in ('images','labels'):
        for s in ('train','val','test'):
            sd=src/d/s; dd=out/d/s; dd.mkdir(parents=True,exist_ok=True)
            if sd.exists():
                for f in sd.iterdir(): shutil.copy2(f,dd/f.name)
    idir,ldir=src/'images'/a.split,src/'labels'/a.split; odir,oldir=out/'images'/a.split,out/'labels'/a.split
    templates=[]
    for ip in idir.iterdir():
        if ip.suffix.lower() not in EXTS:continue
        im=cv2.imread(str(ip)); h,w=im.shape[:2]
        for cls,pts in labels(ldir/f'{ip.stem}.txt'):
            px,py=pts[:,0].mean()*w,pts[:,1].mean()*h; bw=(pts[:,0].max()-pts[:,0].min())*w; bh=(pts[:,1].max()-pts[:,1].min())*h
            templates.append((px/w,py/h,bw/w,bh/h,pts))
    if not templates: raise RuntimeError('No real OBB defect labels found')
    candidates=[]
    for ip in idir.iterdir():
        if ip.suffix.lower() not in EXTS:continue
        if (a.mode=='constrained' or a.restrict_to_masks) and not (Path(a.mask_dir)/f'{ip.stem}.png').exists():continue
        candidates.append(ip)
    random.shuffle(candidates); added=0
    for ip in candidates[:int(len(candidates)*a.ratio)]:
        im=cv2.imread(str(ip)); h,w=im.shape[:2]; entity=mask_for(Path(a.mask_dir)/f'{ip.stem}.png',(h,w)) if a.mode=='constrained' else np.ones((h,w),np.uint8)
        ys,xs=np.where(entity>0); ex1,ey1,ex2,ey2=(xs.min(),ys.min(),xs.max()+1,ys.max()+1) if len(xs) else (0,0,w,h); ew,eh=ex2-ex1,ey2-ey1
        defect=None
        for _ in range(30):
            tx,ty,tw,th,_=random.choice(templates); scale=random.uniform(1-a.size_jitter,1+a.size_jitter); bw,bh=max(3,tw*ew*scale),max(3,th*eh*scale); cx=ex1+(tx+random.uniform(-a.position_jitter,a.position_jitter))*ew; cy=ey1+(ty+random.uniform(-a.position_jitter,a.position_jitter))*eh
            ang=random.uniform(-15,15); poly=cv2.boxPoints(((cx,cy),(bw,bh),ang)).astype(np.int32); dm=np.zeros((h,w),np.uint8); cv2.fillPoly(dm,[poly],255); dm=cv2.bitwise_and(dm,entity*255)
            if a.mode=='ordinary' or cv2.countNonZero(dm)>=.8*max(1,cv2.contourArea(poly.astype(np.float32))): defect=dm;break
        if defect is None:continue
        outim=im.astype(np.float32); outim[defect>0]*=random.uniform(.45,.70); outim=np.clip(outim,0,255).astype(np.uint8); c=np.column_stack(np.where(defect>0)[::-1]).astype(np.float32); name=f'{ip.stem}_synth.jpg'; cv2.imwrite(str(odir/name),outim); rows=labels(ldir/f'{ip.stem}.txt'); text=''.join(str(cls)+' '+' '.join(f'{v:.6f}' for v in pts.reshape(-1))+'\n' for cls,pts in rows); text+='0 '+' '.join(f'{x:.6f}' for x in obb_from_contour(c,w,h))+'\n'; (oldir/f'{Path(name).stem}.txt').write_text(text,encoding='utf-8'); added+=1
    print(f'added={added} mode={a.mode} output={out}')
if __name__=='__main__':main()
