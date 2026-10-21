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

WM-811K 数据集（文件名为 `LSWMD.pkl`），由台积电（TSMC）提供，是目前最大的公开晶圆图缺陷数据集。

| 项目 | 说明 |
|---|---|
| 原始论文 | Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets." *IEEE Transactions on Semiconductor Manufacturing*, 28(1), 1–12. |
| DOI | [10.1109/TSM.2014.2364237](https://doi.org/10.1109/TSM.2014.2364237) |
| 下载地址 | [Kaggle: WM-811K wafer map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)（CC0 公共领域） |
| 总样本数 | 811,457 张晶圆图（47,543 个 lot） |
| 带标签样本 | 约 25,519 张（仅占 ~3.1%，其余为无标签 None 类） |
| 缺陷类别 | 8 类：Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch |
| 图像格式 | 64×64 三值矩阵（0=背景 / 1=良品 / 2=失效） |

> 使用时请自行下载 `LSWMD.pkl` 放置于 `train/data/` 目录，**不要提交至 Git 仓库**（文件较大且为公开数据集）。

## 相关工作与性能对比

以下为基于 WM-811K 数据集的代表性研究，用于定位本项目模型在学术界的位置。

| 方法 | 发表 | 参数量 | Accuracy | Macro-F1 | 部署平台 |
|---|---|---|---|---|---|
| DMC1 (传统方法) | 2019 | — | 94.0% | — | PC |
| DCNN (Saqlain et al.) | 2020 | — | 96.2% | — | PC |
| VGG16 (baseline) | — | ~138M | 92.8% | — | PC |
| DCNN+ECOC-SVM | 2021 | — | 96.4% | — | PC |
| MobileNetV3 (Sensors) | 2023 | ~5.4M | 98.0% | 0.895 | PC |
| 双路径+注意力 (IEEE TSM) | 2025 | 3.17M | 96.8% | — | PC |
| 自编码器增强+CNN (arXiv) | 2024 | — | 98.56% | — | PC |
| **本项目（双输入CNN）** | 2026 | **0.44M** | **89.83%** | **0.869** | **RV1106 NPU** |

### 诚实定位

- **精度差距是预期内的**：多数期刊论文使用 VGG16/ResNet 等大模型（参数量 25M~138M），在 PC GPU 上训练推理。本项目模型仅 442K 参数（小 50~300 倍），且必须满足 RV1106 64MB 内存 + 0.5TOPS NPU 的边缘部署约束。
- **Macro-F1 0.869 接近 MobileNetV3 的 0.895**：在 1/12 参数量下达到相近的宏平均 F1，说明双输入（图像+16维统计特征）架构在小模型上是有效的。
- **少数类仍是瓶颈**：与文献一致，Loc（F1=0.78）和 Scratch（F1=0.83）等少数类精度偏低。Saqlain et al. (2020) 同样报告 Scratch 仅 70.08%、Loc 仅 72.04%，这是 WM-811K 数据集的固有类别不平衡问题。
- **本项目的核心创新不在模型精度，而在系统级**：从训练→量化→边缘NPU部署→Qt上位机→SSE网页看板的全栈闭环，以及统一 edge bin 中间表示和产线全链路自动化。这是上述期刊论文均未覆盖的工程维度。

### 借鉴与创新

| 维度 | 借鉴自文献 | 本项目创新 |
|---|---|---|
| 数据集 | WM-811K (Wu et al. 2015)，标准 8 类划分 | — |
| 模型架构 | CNN 提取晶圆空间纹理（Saqlain 2020 等） | 双输入融合：图像分支 + 16维统计特征分支 |
| 训练方法 | Focal Loss 处理类别不平衡 | SE 注意力 + 特征分支 MLP |
| 量化部署 | INT8 量化是边缘AI常规操作 | 锁定 image/features/output 三组量化参数，喂数契约 |
| 系统设计 | — | 统一 12352B edge bin 格式、多源格式适配层、产线热监控全链路自动化 |
| 监控方案 | — | 64MB 内存板端手写 HTTP/SSE 服务器，替代 WebSocket |
