# 可在 VS/VS Code 中查看和运行的程序脚本

这些文件属于程序代码或启动脚本，可以在 Visual Studio Code 中打开。

## Python 程序

### 训练

- `train_stage1.py`：第一阶段训练
- `train_stage2.py`：第二阶段训练
- `train_obb.py`：OBB 模型训练
- `train_full_baseline.py`：完整基线训练
- `resume_stage1_original.py`：恢复第一阶段训练

### 推理

- `infer_pipeline.py`：两阶段推理流程
- `infer_obb_pipeline.py`：OBB 推理流程
- `infer_one_stage1.py`：第一阶段单独推理
- `infer_stage1_preview_0824.py`：第一阶段预览推理

### 评估、导出和可视化

- `eval_model.py`：评估模型
- `export_onnx.py`：导出 ONNX 模型
- `visualize_obb_labels.py`：可视化 OBB 标注
- `batch_obb_preview.py`：批量生成 OBB 预览图
- `auto_label_insulator.py`：自动标注绝缘子
- `launch_val_labelimg.py`：启动验证集标注工具

## PowerShell 启动脚本

- `run_obb_gpu_d.ps1`
- `run_obb_finetune_50_gpu_d.ps1`
- `run_obb_finetune_100_gpu_d.ps1`
- `run_stage1_original_gpu_100.ps1`
- `run_stage1_original_finetune100.ps1`
- `run_two_stage_100_gpu_d.ps1`

这些脚本主要用于启动训练任务。可以在 VS Code 的终端中执行，但需要先激活项目的 `.venv` 环境。

## scripts/ 目录

`scripts/` 中也是程序脚本，主要负责数据处理、标注转换、数据集切分、检查和合成数据生成。

## LabelImgApp/ 目录

这是标注工具的程序源码，不是实验数据。

## 不是程序脚本的文件

- `data/*.yaml`、`configs/*.yaml`：配置文件
- `obb_dataset/`、`datasets/`：实验数据和标注
- `runs/`、`outputs/`、`test_results/`：训练和测试结果
- `*.pt`：模型权重
- `*.jpg`、`*.png`：图片或可视化结果
- `*.log`、`*.err.log`：运行日志

## 在 VS Code 中打开

在项目目录 `D:\GeoROI-YOLO` 中打开终端，执行：

```powershell
code .
```

建议先打开和修改 `.py` 文件；运行训练前再确认对应的 `data/*.yaml` 配置和模型权重文件。
