# train-A 完整训练程序包

## 主要程序
- `train_stage1.py`：一阶段绝缘子检测训练
- `train_stage2.py`：二阶段缺陷检测训练
- `train_obb.py`：OBB 旋转框训练
- `infer_pipeline.py`：一阶段+二阶段完整 pipeline 推理
- `infer_obb_pipeline.py`：OBB pipeline 推理
- `eval_model.py`：模型评估
- `export_onnx.py`：导出 ONNX

## 使用方式
在 VS Code 终端进入本目录后，先激活原项目环境，或安装 `requirements.txt`：

```powershell
cd D:\GeoROI-YOLO-9.12\train-A
python train_stage1.py --help
python train_stage2.py --help
python infer_pipeline.py --help
```

训练权重位于 `runs\...\weights\best.pt` 和 `last.pt`。基础模型位于根目录的 `yolo*.pt`。数据集路径仍按 `data\*.yaml` 中的配置执行；如数据集不在本目录，请修改对应 YAML 的 `path`。
