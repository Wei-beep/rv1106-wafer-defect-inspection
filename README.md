# RV1106 Wafer Defect Inspection System

> 基于瑞芯微 RV1106（ARM Cortex-A7 + 0.5TOPS NPU）的晶圆缺陷全栈检测系统

## 项目简介

本项目实现了一套从深度学习模型训练到边缘端部署的完整晶圆缺陷检测系统。采用**双输入 CNN**（64×64×3 晶圆图 + 16 维统计特征）对 8 类典型缺陷模式进行分类，经 INT8 量化后在 RV1106 NPU 上单帧推理 **2.12 ms**，板端准确率 **92.5%**。

系统包含三部分：

- **板端 C 一体化推理服务**（`wafer_direct.c`）：单进程 5 线程，同时提供 TCP 推理、HTTP/SSE 网页看板、UDP 设备发现、目录热监控直解；
- **Qt 上位机**（C++）：研发调试 / 产线直解双模式，支持 STDF/BIN/PNG 多源格式自动适配、文件夹热监控、批次评价与 CSV 导出；
- **SSE 实时网页看板**：纯 HTML/JS 单文件，板端 HTTP 直接服务，无需额外 Web 服务器。

## 技术栈

| 层级 | 技术 |
|---|---|
| 板端语言 | C（pthread 多线程） |
| NPU 推理 | RKNN Runtime 1.6.0（librknnmrt） |
| 网络协议 | TCP 8080（推理）/ HTTP+SSE 8081（看板+上传）/ UDP 8090（发现） |
| 上位机 | C++ / Qt5 / QFileSystemWatcher / QSettings |
| 模型链路 | PyTorch → ONNX → RKNN INT8 量化 |
| 交叉编译 | `arm-rockchip830-linux-uclibcgnueabihf-gcc`（uclibc） |
| 硬件平台 | Luckfox Pico（RV1106，64 MB DDR2，Buildroot Linux） |

## 功能特性

- ✅ ATE 多源数据适配：STDF / BIN / PNG → 统一 12352 B edge bin，板端零修改
- ✅ 双输入融合分类：图像空间纹理 + 16 维统计特征，8 类缺陷准确率 92.5%
- ✅ NPU INT8 量化部署：单帧 2.12 ms，较 CPU（17.4 ms）加速 8.3 倍
- ✅ 板端 5 线程一体化服务，64 MB 内存稳定运行，峰值 < 10 MB
- ✅ UDP 广播自动发现板子 IP，产线免手动配置
- ✅ SSE 实时推送看板，浏览器原生重连，板端实现成本极低
- ✅ 产线文件夹热监控：文件写入完成自动转换→上传→推理→上屏，全链路无人值守
- ✅ Buildroot 开机自启（S99 脚本），产线零 SSH 操作
- ✅ Qt 上位机中英双语、研发/产线双模式一键切换

## 性能指标

| 指标 | 数值 |
|---|---|
| NPU 单帧推理 | 2.12 ms |
| CPU 推理（对比） | 17.4 ms |
| NPU 加速比 | 8.3× |
| 板端准确率 | 92.5%（185/200） |
| 板端内存峰值 | < 10 MB |
| 支持输入格式 | STDF / BIN / PNG |
| 统一中间格式 | 12352 B edge bin（12288 B RGB + 64 B 特征） |

## 缺陷类别

Center · Donut · Edge-Loc · Edge-Ring · Loc · Near-full · Random · Scratch

## 系统架构

```
产线 STDF/BIN/PNG
       │
       ▼
┌─────────────────────┐
│  Qt 上位机 (C++)     │  格式适配 WaferConverter
│  ├─ 研发模式: TCP直连 │  文件夹热监控 QFileSystemWatcher
│  └─ 产线模式: HTTP上传│  UDP发现 8090
└─────────┬───────────┘
          │ TCP 8080 / HTTP 8081
          ▼
┌─────────────────────────────────────┐
│  RV1106 板端 wafer_direct (C, 5线程) │
│  ├─ direct_loop: 目录监控 incoming/  │
│  ├─ tcp_loop:    TCP 推理 8080       │
│  ├─ http_loop:   HTTP+SSE 8081       │
│  ├─ sse_heartbeat: SSE 心跳          │
│  └─ udp_broadcast: UDP 发现 8090     │
│         │                            │
│         ▼                            │
│  RKNN NPU 推理 (2.12ms/帧)           │
└─────────┬───────────────────────────┘
          │ SSE 推送
          ▼
   网页实时看板 (dashboard.html)
```

## 目录结构

```
rv1106-wafer-defect-inspection/
├── README.md
├── LICENSE
├── .gitignore
├── docs/
│   ├── architecture.md        # 架构与协议说明（端口/edge bin/TCP协议/SSE选型）
│   ├── build.md               # 交叉编译与Qt编译步骤
│   ├── deploy.md              # 板端部署与开机自启
│   ├── training_results.md    # 模型训练性能数据
│   ├── rknn_quantization.md   # RKNN INT8量化转换步骤与喂数契约
│   └── pitfalls.md            # 踩坑记录与根因分析
├── train/                     # 模型训练（PyTorch）
│   ├── train.py               # 完整训练脚本（Focal Loss+早停+ONNX导出）
│   ├── wm811k_dataset.py      # 数据集加载+模型定义+16维特征提取
│   └── requirements.txt       # Python 依赖
├── board/                     # 板端 C 服务
│   ├── wafer_direct.c         # 一体化服务（目录直解+TCP+HTTP/SSE+UDP），当前主力
│   └── wafer_edge_server.c    # 旧版 TCP+SSE 服务（已废弃，保留参考）
├── host/                      # Qt 上位机
│   ├── mainwindow.cpp / .h
│   ├── devicepanel.cpp / .h
│   ├── waferconverter.cpp / .h
│   └── WaferInspector.pro
├── web/
│   └── dashboard.html         # SSE 实时看板（独立参考版，板端实际内嵌于 wafer_direct.c）
├── tools/
│   ├── any_to_edge.py         # 离线格式转换（BIN/PNG/STDF）
│   └── bin_to_edge.py
├── scripts/
│   ├── cross_compile.sh       # 交叉编译脚本
│   └── S99waferdirect         # Buildroot 开机自启
└── model/
    ├── wafer_pc_int8.rknn     # INT8 量化模型（Git LFS）
    └── wafer_model.onnx       # ONNX 导出模型（Git LFS，可选）
```

## 模型训练

模型基于 WM-811K 公开数据集训练，采用双输入 CNN（图像分支 + 16 维手工特征分支 + SE 通道注意力），Focal Loss 处理类别不均衡。

```bash
cd train
pip install -r requirements.txt
# 将 LSWMD.pkl 放入 train/data/ 目录（需自行下载，勿提交Git）
python train.py --data ./data/LSWMD.pkl --epochs 60 --batch_size 64 --gamma 1.5
```

训练完成后输出：最佳模型 `best_wafer.pt`、ONNX 模型 `wafer_model.onnx`、特征归一化参数 `feature_norm.npz`、混淆矩阵与训练曲线。

| 指标 | 值 |
|---|---|
| 测试集 Accuracy | 89.83% |
| 测试集 Macro-F1 | 0.8689 |
| 模型参数量 | 442,616 |
| ONNX 模型大小 | 1.70 MB |
| RKNN INT8 板端准确率 | 92.5%（200张独立子集） |
| NPU 单帧推理 | 2.12 ms |

详见 [docs/training_results.md](docs/training_results.md)。

## 快速开始

### 1. 交叉编译板端服务

```bash
# 在 x86 Ubuntu 主机上
arm-rockchip830-linux-uclibcgnueabihf-gcc board/wafer_direct.c -o wafer_direct \
  -I./ -L$HOME/.local/lib/python3.8/site-packages/rknn/3rdparty/\
platform-tools/runtime/Linux/librknn_api/armhf-uclibc \
  -lrknnmrt -lm -lpthread -O2
```

> 必须使用 uclibc 版本的交叉编译器和 librknnmrt，glibc 版本会因 ABI 不兼容导致段错误。

### 2. 部署到板子

```bash
scp wafer_direct model/wafer_pc_int8.rknn root@<板子IP>:/root/wafer_test/8_class/
ssh root@<板子IP>
cd /root/wafer_test/8_class
chmod +x wafer_direct
export LD_LIBRARY_PATH=/oem/usr/lib:./
mkdir -p incoming processed
./wafer_direct wafer_pc_int8.rknn &
```

启动成功后应看到：

```
[INFO] RKNN ready, in0=image in1=features out=logits
[INFO] ready. TCP=:8080(edge NPU) + HTTP=:8081(direct+web) + UDP=:8090(discovery)
```

### 3. 访问网页看板

浏览器打开 `http://<板子IP>:8081`，即可看到实时推理流与性能指标。

### 4. 编译 Qt 上位机

```bash
cd host
qmake WaferInspector.pro
make -j4
```

## 关键技术细节

### 双输入量化（核心踩坑点）

- **图像分支**：必须喂 `uint8` 原始 RGB NHWC 像素（0~255），由 NPU 内部融合归一化（scale=0.003922, zp=-128）。**禁止手动 int8 量化**，否则双重量化导致精度暴跌。
- **特征分支**：手动 int8 量化 `q = round(x / 0.099453 + 13)`。
- **输出反量化**：`float = (int8 - zp) * scale`，output scale=0.255337, zp=3。

### 统一 edge bin 格式

```
12352 B = 12288 B (64×64×3 uint8 RGB NHWC) + 64 B (16×float32 特征, Z-Score 归一化)
```

任何测试机格式只需适配一个解析器，转换为 edge bin 即可被板端消费。

### 网页看板说明

板端实际运行的看板 HTML 以字符串常量形式**内嵌在 `wafer_direct.c` 的 `http_loop` 中**，这样单个二进制文件即可自带 Web 服务，不依赖板端文件系统。`web/dashboard.html` 为独立参考版本，功能与板端一致（SSE 实时流、性能卡片、双语切换、断线重连），便于阅读、复用和二次开发。如需让板端读取外部文件，可将 `http_loop` 中写死的字符串替换为 `fopen("dashboard.html")` 读文件后发送。

### TCP 推理协议

```
Qt → 板: 12356 B = "RFNS"(4) + img(12288) + feat(64)
板 → Qt: 44 B    = "RFNR"(4) + pred(int32) + conf(float32) + 8×logits(float32)
```

## 数据集与相关工作

### 数据集

本项目使用 **WM-811K**（文件名 `LSWMD.pkl`），由台积电（TSMC）提供，是目前最大的公开晶圆图缺陷数据集。

- **原始论文**：Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets." *IEEE Transactions on Semiconductor Manufacturing*, 28(1), 1–12. [DOI](https://doi.org/10.1109/TSM.2014.2364237)
- **下载地址**：[Kaggle: WM-811K wafer map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)（CC0 公共领域）
- **规模**：811,457 张晶圆图，其中带标签 25,519 张，8 类缺陷模式
- **使用方式**：自行下载 `LSWMD.pkl` 放置于 `train/data/`，**不要提交至 Git**

### 与期刊论文的性能对比

| 方法 | 参数量 | Accuracy | Macro-F1 | 平台 |
|---|---|---|---|---|
| VGG16 (baseline) | ~138M | 92.8% | — | PC |
| MobileNetV3 (Sensors 2023) | ~5.4M | 98.0% | 0.895 | PC |
| 双路径+注意力 (IEEE TSM 2025) | 3.17M | 96.8% | — | PC |
| **本项目** | **0.44M** | **89.83%** | **0.869** | **RV1106 NPU** |

> 本项目模型参数量仅为 VGG16 的 1/312、MobileNetV3 的 1/12，在 64MB 内存边缘 NPU 上实现 2.12ms/帧推理。精度差距源于模型规模约束，Macro-F1 已接近 MobileNetV3。核心创新在全栈边缘部署系统，而非 SOTA 精度。详见 [docs/training_results.md](docs/training_results.md)。

## 创新点

- **双输入融合模型**：图像空间纹理 + 16 维统计特征双分支 CNN，较单图像模型提升分类精度
- **统一中间表示**：12352B edge bin 解耦数据源与推理端，新测试机格式只需适配一个解析器
- **多源格式适配层**：STDF/BIN/PNG → edge bin，Qt C++ 实现无 Python 依赖，策略模式可扩展
- **产线全链路自动化**：文件夹热监控 + 写入稳定性检测 + 自动转换上传 + NPU 推理 + SSE 实时上屏

## 踩坑记录

开发过程中遇到的 8 个关键问题及根因分析，详见 [docs/pitfalls.md](docs/pitfalls.md)。包括：图像双重量化精度暴跌、uclibc/glibc ABI 段错误、E2E 时延 773ms→2ms 计时修正、Qt 图层重叠等。

## 未来规划

### 短期
- 真实产线 STDF 格式适配（多 Part / 多 Test Number 映射）
- TTA 测试时增强 + 困难样本挖掘，精度目标 95%+
- 板端多 batch 推理 + 输入预取，吞吐量提升 2-3 倍

### 中长期
- 多板集群调度与负载均衡
- OTA 远程更新 RKNN 模型
- MES 系统对接（OPC UA / REST API）
- 联邦学习（多产线数据联合训练，数据不出厂）

## 许可证

MIT License
