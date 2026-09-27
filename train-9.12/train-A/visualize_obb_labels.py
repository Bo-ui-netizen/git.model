from pathlib import Path
import argparse
import cv2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--image', required=True); p.add_argument('--out', default='outputs/obb_label_preview.jpg'); a=p.parse_args()
    image_path=Path(a.image); im=cv2.imread(str(image_path)); label=image_path.parents[1]/'labels'/image_path.parent.name/(image_path.stem+'.txt')
    h,w=im.shape[:2]
    if label.exists():
        for line in label.read_text(encoding='utf-8').splitlines():
            v=line.split();
            if len(v)!=9: continue
            pts=[]
            for i in range(1,9,2): pts.append([round(float(v[i])*w), round(float(v[i+1])*h)])
            pts=cv2.UMat(cv2.array(pts)) if False else __import__('numpy').array(pts,dtype='int32').reshape((-1,1,2))
            cv2.polylines(im,[pts],True,(0,0,255),3)
            x,y=pts[0,0]; cv2.putText(im,'defect',(int(x),max(25,int(y)-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(a.out,im); print(Path(a.out).resolve())
if __name__=='__main__': main()
