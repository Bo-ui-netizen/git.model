# 项目文件分类清单

## 1. 可运行程序（源码/脚本）

根目录 Python：`auto_label_insulator.py`、`batch_obb_preview.py`、`eval_model.py`、`export_onnx.py`、`infer_*.py`、`train_*.py`、`visualize_obb_labels.py`。

根目录 PowerShell/VBS：`run_*.ps1`、`launch_labelimg.vbs`、`bootstrap.sh`。

辅助程序目录：`scripts/`（数据转换、切分、检查、合成等脚本）、`LabelImgApp/`（标注工具源码）。

## 2. 配置与说明

- `configs/`：项目配置
- `data/*.yaml`：数据集配置
- `docs/`、`README.md`、`新手运行说明.md`、`models/README_P2.md`：说明文档
- `requirements.txt`：Python 依赖

## 3. 实验数据与标注

- `datasets/`：数据集总目录（当前顶层可能为空，需检查子目录）
- `obb_dataset/`：OBB 训练/验证/测试数据及标签
- `obb_dataset_synth_constrained/`、`obb_dataset_synth_ordinary/`：合成实验数据集
- `data/`：数据集 YAML 配置（配置文件，不是图片数据）
- `empty_val_labels_backup_20260813/`、`label_backup_*`：标注备份
- `mask_review/`：掩膜/标注复核数据

## 4. 模型与训练产物

- `*.pt`：预训练/基础模型权重（`yolo11n.pt`、`yolo11n-obb.pt`、`yolo26n.pt`）
- `runs/`：训练运行日志、错误日志及可能的训练结果
- `outputs/`：训练日志、推理输出和其他实验输出
- `test_results/`：测试预测图、混淆矩阵、PR/F1 曲线等评估结果
- `stage_preview/`、`codex_images/`：阶段预览图和临时展示图

## 5. 备份、压缩包与交付文件

- `backups/`：备份目录
- `GeoROI-YOLO_程序版_无数据.zip`：不含数据的程序交付包
- `.zip`：大型压缩包，建议确认内容后移入归档区

## 6. 环境、缓存与临时文件（一般不属于项目成果）

- `.venv/`：Python 虚拟环境
- `__pycache__/`：Python 缓存
- `.cache/`、`.cache_d/`、`.torch_cache/`、`.yolo_config/`、`.labelme_local/`、`.tmp/`：缓存/临时目录
- `.idea/`：IDE 工程设置
- `*.log`、`*.err.log`：运行日志（有实验记录价值的保留在 `runs/`/`outputs/`，零字节测试日志可删除）
- `write_test.txt`：测试文件

## 建议的整理目标

保留项目根目录的入口脚本、README、requirements 和三个核心目录：`src/`（程序）、`data/`（数据）、`experiments/`（runs/outputs/test_results）。但由于现有脚本互相引用根目录路径，建议先按本清单建立分类索引，再分批移动并同步修改路径。

### 当前最安全的第一步

先清理 `__pycache__`、缓存目录和 `write_test.txt`；保留数据集、备份、模型权重和实验结果原位不动，避免破坏脚本路径。
