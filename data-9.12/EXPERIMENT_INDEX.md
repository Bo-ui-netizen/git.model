# 实验数据与结果

## 第一阶段训练
`runs/stage1_insulator_gpu100`、`runs/stage1_original_gpu100`、`runs/stage1_original_finetune100`、`runs/stage1_paper_gpu`，以及 `logs` 中对应日志。

## OBB/第二阶段训练
`runs/stage2_*`、`runs/obb`、`runs/demo_obb`，其中包含权重、指标和训练曲线。

## 合成增强对比
`runs/ablation_*`、`obb_dataset_synth_*`。

## 测试与输出
`test_results`、`outputs`、`stage_preview`。

每个训练目录下的 `weights/best.pt` 是最佳模型，`weights/last.pt` 是最后一轮模型。
