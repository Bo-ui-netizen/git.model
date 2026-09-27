# 无人机缺陷检测：Geo-ROI-YOLO

这是一个面向无人机电力线路绝缘子缺陷检测的可执行研究工程。它基于 CPLID 数据集和 Ultralytics YOLO，包含数据转换、ROI 裁剪、缺陷合成增强、两阶段训练、几何先验后处理、推理可视化和 ONNX 导出。

## 目录

```text
无人机缺陷检测/
├── data/                    # Ultralytics 数据集 YAML
├── datasets/                # 数据集目录，下载后放这里
├── scripts/                 # 数据处理、增强、几何后处理
├── models/                  # P2 小目标头说明/模型配置预留
├── docs/                    # 实验执行方案
├── train_full_baseline.py   # 全图 baseline
├── train_stage1.py          # 第一阶段：绝缘子检测
├── train_stage2.py          # 第二阶段：ROI 缺陷检测
├── infer_pipeline.py        # 两阶段 + 几何先验推理
├── export_onnx.py           # ONNX 导出
├── eval_model.py            # 验证指标
├── bootstrap.sh             # 创建 Python 环境
└── requirements.txt
```

## 1. 环境安装

```bash
cd ~/Desktop/无人机缺陷检测
bash bootstrap.sh
source .venv/bin/activate
```

Apple Silicon 可以训练时把 `--device cpu` 改成 `--device mps`。

## 2. 下载 CPLID 数据集

```bash
cd ~/Desktop/无人机缺陷检测
git clone https://github.com/InsulatorData/InsulatorDataSet.git datasets/InsulatorDataSet
```

如果 GitHub 下载慢，可以手动下载 zip，解压后保证路径为：

```text
datasets/InsulatorDataSet
```

## 3. VOC XML 转 YOLO

```bash
python scripts/voc2yolo_cplid.py \
  --raw datasets/InsulatorDataSet \
  --out datasets/CPLID_YOLO \
  --val-ratio 0.2
```

输出两份数据：

```text
datasets/CPLID_YOLO/full       # class 0 insulator, class 1 defect
datasets/CPLID_YOLO/insulator  # class 0 insulator only
```

## 4. 训练全图 baseline

```bash
python train_full_baseline.py \
  --data data/cplid_full.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 80 \
  --batch 4 \
  --device cpu
```

结果默认保存到：

```text
runs/detect/baseline_full_yolo11n
```

## 5. 训练第一阶段绝缘子检测器

```bash
python train_stage1.py \
  --data data/cplid_insulator.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 80 \
  --batch 4 \
  --device cpu
```

## 6. 生成 ROI 缺陷数据集

```bash
python scripts/make_roi_dataset.py \
  --src datasets/CPLID_YOLO/full \
  --out datasets/CPLID_ROI \
  --padding 0.30 \
  --size 768
```

第二阶段只有一个类别：

```text
class 0: defect
```

没有缺陷的 ROI 会生成空标签，用作负样本。

## 7. 加入缺陷合成增强

```bash
python scripts/synth_defects.py \
  --src datasets/CPLID_ROI \
  --split train \
  --ratio 0.5
```

`--ratio 0.5` 表示从无缺陷 ROI 中抽取一半生成合成缺陷。普通模式使用整张 ROI 随机生成；约束模式需要 `masks/<split>/<ROI同名>.png` 绝缘子实体掩膜，并从真实标签统计尺寸和相对位置。三组对照可运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_synth_ablation.ps1 -Device 0 -Epochs 120
```

三组实验分别为：不使用合成增强、普通随机合成、实体约束且与真实缺陷统计对齐的合成。除增强策略外，模型、数据划分、训练轮数、批大小和设备保持一致；结果目录为 `runs/ablation_no_synth`、`runs/ablation_ordinary_synth`、`runs/ablation_constrained_stats`。

## 8. 训练第二阶段 ROI 缺陷检测器

```bash
python train_stage2.py \
  --data data/cplid_defect_roi.yaml \
  --model yolo11n.pt \
  --imgsz 768 \
  --epochs 120 \
  --batch 2 \
  --device cpu
```

## 9. 两阶段推理与可视化

```bash
python infer_pipeline.py \
  --image datasets/CPLID_YOLO/full/images/val/your_image.jpg \
  --stage1 runs/detect/stage1_insulator/weights/best.pt \
  --stage2 runs/detect/stage2_defect_roi/weights/best.pt \
  --out outputs/infer \
  --device cpu
```

输出包含 ROI 裁剪图和最终检测图：

```text
outputs/infer/*_geo_roi_result.jpg
```

## 10. 验证和导出

```bash
python eval_model.py \
  --model runs/detect/stage2_defect_roi/weights/best.pt \
  --data data/cplid_defect_roi.yaml \
  --imgsz 768 \
  --device cpu

python export_onnx.py \
  --model runs/detect/stage2_defect_roi/weights/best.pt \
  --imgsz 768
```

## 11. 论文实验建议

按以下顺序填表：

1. `YOLO11n-full`：全图 baseline。
2. `YOLO11n-full-high`：全图 1024 输入。
3. `ROI-YOLO`：ROI 裁剪后第二阶段检测。
4. `ROI-YOLO + Synth`：加入缺陷合成增强。
5. `Ours + Geo`：推理时启用几何先验分数融合。
6. `ROI-YOLO-P2`：后续加入 P2 小目标头。

详细实验方案见 `docs/EXECUTION_PLAN.md`。
