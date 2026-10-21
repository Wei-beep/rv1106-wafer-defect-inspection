# RKNN 模型量化与转换 / RKNN Quantization & Conversion

本文档记录从 ONNX 到 RKNN INT8 量化模型的转换过程。仓库中暂未包含量化脚本（依赖 RKNN Toolkit 环境），以下为操作步骤与关键参数。

## 环境要求

- Python 3.8
- RKNN Toolkit（`rknn-toolkit2` 或 `rknn-toolkit-lite2`）
- ONNX 模型：`model/wafer_model.onnx`（训练脚本导出）

## 转换步骤

```python
from rknn.api import RKNN

rknn = RKNN(verbose=True)

# 1. 配置量化参数
rknn.config(
    mean_values=[[0, 0, 0]],       # image 分支不做均值偏移（NPU内部融合归一化）
    std_values=[[255, 255, 255]],  # image 分支归一化 scale=1/255=0.003922
    target_platform='rv1106',
    quantized_dtype='asymmetric_quantized-8',
)

# 2. 加载 ONNX 模型
rknn.load_onnx(model='wafer_model.onnx')

# 3. 构建（量化）
#    dataset.txt 每行是一个校准样本的图像路径（用于统计量化参数）
rknn.build(do_quantization=True, dataset='dataset.txt')

# 4. 导出 RKNN 模型
rknn.export_rknn('wafer_pc_int8.rknn')

# 5. 精度验证（可选）
rknn.accuracy_analysis(inputs=['calibration_sample.npy'])

rknn.release()
```

## 关键量化参数

| 张量 | scale | zero_point | 量化方式 |
|---|---|---|---|
| image 输入 | 0.003922 | -128 | NPU 内部融合归一化（std=255） |
| features 输入 | 0.099453 | 13 | 手动 int8 量化 |
| output 输出 | 0.255337 | 3 | NPU 反量化 |

## 板端喂数契约（核心，勿改）

- **image 分支**：喂 `uint8` 原始 RGB NHWC 像素（0~255），**禁止手动 int8 量化**。NPU 内部已融合归一化，手动量化会导致双重量化、精度暴跌。
- **features 分支**：手动 int8 量化 `q = round(x / 0.099453 + 13)`，其中 x 是 Z-Score 归一化后的 float32 特征。
- **output**：获取 int8 后手动反量化 `float = (int8 - 3) * 0.255337`。

## 校准数据集

量化校准使用训练集中的代表性样本（建议 100~500 张），`dataset.txt` 格式：

```
calib/img_001.bin
calib/img_002.bin
...
```

每个文件为 64×64×3 uint8 RGB 原始像素（NHWC 布局）。

## 精度对比

| 模型 | 测试集 | Accuracy |
|---|---|---|
| PyTorch float32 | WM-811K 测试集（3834张） | 89.83% |
| RKNN int8（板端） | 独立子集（200张） | 92.5% |

> 注：两者测试集不同，板端 200 张为独立采样子集，不存在精度矛盾。同测试集上量化损失 < 2%。

## 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| 板端精度暴跌（<50%） | image 分支手动做了 int8 量化 | 改喂原始 uint8 像素 |
| rknn_inputs_set 返回 -5 | 轻量库对双输入支持有 bug | 用标准 rknn_input 结构体 API |
| 输出改 float32 被拒 | 量化模型输出类型固定 int8 | 获取 int8 后手动反量化 |
| 交叉编译段错误 | 用了 glibc 编译器 | 必须用 uclibc 版 arm-rockchip830 |
