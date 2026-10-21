# 模型训练结果 / Training Results

## 训练配置

| 项目 | 值 |
|---|---|
| 数据集 | WM-811K（LSWMD.pkl） |
| 带标签样本 | 25,519 张（总 811,457 张） |
| 划分 | 训练 17,861 / 验证 3,824 / 测试 3,834 |
| 模型 | 双输入 CNN（图像分支 + 16维特征分支 + SE注意力） |
| 参数量 | 442,616 |
| 损失函数 | Focal Loss（gamma=1.5，带类别权重） |
| 优化器 | AdamW（lr=3e-4, weight_decay=1e-4） |
| 学习率调度 | ReduceLROnPlateau（factor=0.5, patience=5） |
| Early Stopping | patience=15，监控 val_macro_f1 |
| 最佳轮次 | Epoch 31 |
| 训练硬件 | NVIDIA RTX 4060 Laptop GPU |
| 单轮耗时 | ~34s/epoch |

## 测试集性能

| 指标 | 值 |
|---|---|
| Accuracy | **89.83%** |
| Macro-F1 | **0.8689** |
| Macro-Precision | 0.8592 |
| Macro-Recall | 0.8849 |

### 各类别 F1 / Recall（测试集）

| 类别 | F1 | Recall | 支持样本数 |
|---|---|---|---|
| Center | 0.9375 | 0.9659 | 645 |
| Donut | 0.8187 | 0.8333 | 84 |
| Edge-Loc | 0.8490 | 0.9024 | 779 |
| Edge-Ring | 0.9644 | 0.9428 | 1452 |
| Loc | 0.7787 | 0.7296 | 540 |
| Random | 0.9091 | 0.9160 | 131 |
| Scratch | 0.8256 | 0.7889 | 180 |
| Near-full | 0.8679 | 1.0000 | 23 |

## ONNX 导出与校验

| 项目 | 值 |
|---|---|
| ONNX 模型大小 | 1.70 MB |
| 输入 | image (1×3×64×64), features (1×16) |
| 输出 | logits (1×8) |
| PyTorch ↔ ONNX 最大误差 | 9.49e-04 |
| 校验结果 | 通过（max_diff < 1e-2） |

## RKNN INT8 量化后板端性能

| 项目 | 值 |
|---|---|
| 板端测试集 | 200 张（独立子集） |
| 板端准确率 | **92.5%**（185/200） |
| NPU 单帧推理 | 2.12 ms |
| CPU 推理（对比） | 17.4 ms |
| NPU 加速比 | 8.3× |

> 注：板端 92.5% 与测试集 89.83% 的差异源于测试集不同——板端验证使用的是 200 张独立采样子集，两者口径不同，不存在矛盾。

## 数据集来源

WM-811K 数据集（Wu et al., 2014），原始文件 LSWMD.pkl。
- 论文：*Wu, M. et al. "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets." IEEE Transactions on Semiconductor Manufacturing, 2014.*
- 请勿将 LSWMD.pkl 提交至 Git 仓库（文件较大且为公开数据集），使用时请自行下载并放置于 `train/data/` 目录。
