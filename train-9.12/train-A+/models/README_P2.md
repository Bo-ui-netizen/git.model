# YOLO11n-P2 小目标头说明

当前工程默认使用 `yolo11n.pt`，先保证 CPLID 转换、ROI 裁剪、合成增强、两阶段推理和几何后处理完整跑通。

P2 小目标头建议作为第二轮实验加入：

1. 安装并确认 Ultralytics 版本后，找到其对应的 `yolo11n.yaml`。
2. 复制到本目录，命名为 `yolo11n-p2-defect.yaml`。
3. 在 neck 中引入 stride=4 的浅层特征，并让 Detect 层使用 P2/P3/P4。
4. 使用同一份 `data/cplid_defect_roi.yaml` 训练，对比 ROI-YOLO 与 ROI-YOLO-P2。

论文写作时可以把 P2 作为消融项，而不是第一阶段必须完成的依赖。这样最小可行版本更稳。
