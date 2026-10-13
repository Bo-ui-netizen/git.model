from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO
from scripts.common import enlarge_box, ensure_dir, read_image

def result_boxes(result):
    if result.boxes is None:
        return []
    out = []
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        out.append((x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
    return out

def draw_box(im, box, color, label):
    x1, y1, x2, y2 = map(lambda v: int(round(v)), box[:4])
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    cv2.putText(im, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)

def draw_obb(im, corners, color=(0, 0, 255), label=None):
    pts = corners.astype('int32').reshape((-1, 1, 2))
    cv2.polylines(im, [pts], True, color, 2)
    if label:
        x, y = pts[0, 0]
        cv2.putText(im, label, (int(x), max(20, int(y)-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--stage1', default='runs/detect/stage1_insulator/weights/best.pt')
    p.add_argument('--stage2', required=True, help='OBB模型权重，如 yolo11n-obb.pt')
    p.add_argument('--out', default='outputs/infer')
    p.add_argument('--device', default='0', help='GPU 编号；RTX 4060 通常为 0')
    p.add_argument('--padding', type=float, default=.30)
    # Keep inference aligned with the high-resolution OBB training default.
    p.add_argument('--roi-size', type=int, default=1024)
    p.add_argument('--conf1', type=float, default=.25)
    p.add_argument('--conf2', type=float, default=.20)
    a = p.parse_args()
    image_path = Path(a.image)
    image = read_image(image_path); h, w = image.shape[:2]
    stage1_vis = image.copy()
    stage2_vis = image.copy()
    vis = image.copy()
    out = ensure_dir(a.out)
    s1, s2 = YOLO(a.stage1), YOLO(a.stage2)
    ins = result_boxes(s1.predict(image, imgsz=640, conf=a.conf1, device=a.device, verbose=False)[0])
    for ib in ins:
        draw_box(stage1_vis, ib, (0, 180, 255), f'insulator {ib[4]:.2f}')
    stage1_path = out / f'{image_path.stem}_stage1_insulators.jpg'
    cv2.imwrite(str(stage1_path), stage1_vis)
    n = 0
    # 若第一阶段没有找到绝缘子，直接在整图运行 OBB，保证可以看到结果。
    if not ins:
        r = s2.predict(image, imgsz=a.roi_size, conf=a.conf2, device=a.device, verbose=False)[0]
        if r.obb is not None:
            for poly, conf, cls in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy()):
                label = f'defect {int(cls)} {float(conf):.2f}'
                draw_obb(stage2_vis, poly, label=label)
                draw_obb(vis, poly, label=label)
                n += 1
        stage2_path = out / f'{image_path.stem}_stage2_defects.jpg'
        path = out / f'{image_path.stem}_obb_result.jpg'
        cv2.imwrite(str(stage2_path), stage2_vis); cv2.imwrite(str(path), vis)
        print(f'Saved stage1={stage1_path}, stage2={stage2_path}, combined={path}; insulators=0, obb_defects={n} (full-image fallback)')
        return
    for i, ib in enumerate(ins):
        x1, y1, x2, y2 = enlarge_box(ib[:4], w, h, a.padding); crop = image[y1:y2, x1:x2]
        if crop.size == 0: continue
        roi = cv2.resize(crop, (a.roi_size, a.roi_size)); r = s2.predict(roi, imgsz=a.roi_size, conf=a.conf2, device=a.device, verbose=False)[0]
        if r.obb is not None:
            sx, sy = (x2-x1)/a.roi_size, (y2-y1)/a.roi_size
            for poly, conf, cls in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy()):
                poly[:,0] = x1 + poly[:,0]*sx; poly[:,1] = y1 + poly[:,1]*sy
                label = f'defect {int(cls)} {float(conf):.2f}'
                draw_obb(stage2_vis, poly, label=label)
                draw_obb(vis, poly, label=label); n += 1
        draw_box(vis, (x1,y1,x2,y2), (0,180,255), f'insulator {ib[4]:.2f}')
        cv2.imwrite(str(out / f'{Path(a.image).stem}_roi{i:02d}.jpg'), roi)
    stage2_path = out / f'{image_path.stem}_stage2_defects.jpg'
    path = out / f'{image_path.stem}_obb_result.jpg'
    cv2.imwrite(str(stage2_path), stage2_vis); cv2.imwrite(str(path), vis)
    print(f'Saved stage1={stage1_path}, stage2={stage2_path}, combined={path}; insulators={len(ins)}, obb_defects={n}')

if __name__ == '__main__': main()
