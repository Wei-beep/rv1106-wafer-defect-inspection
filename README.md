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
│   ├── architecture.md        # 架构与协议说明
│   ├── build.md               # 交叉编译步骤
│   └── deploy.md              # 板端部署与开机自启
├── board/                     # 板端 C 服务
│   ├── wafer_direct.c         # 一体化服务（目录直解+TCP+HTTP/SSE+UDP）
│   └── wafer_edge_server.c    # 旧版 TCP+SSE 服务
├── host/                      # Qt 上位机
│   ├── mainwindow.cpp / .h
│   ├── devicepanel.cpp / .h
│   ├── waferconverter.cpp / .h
│   └── wafer_direct.pro
├── web/
│   └── dashboard.html         # SSE 实时看板（独立参考版，板端实际内嵌于 wafer_direct.c）
├── tools/
│   ├── any_to_edge.py         # 离线格式转换（BIN/PNG/STDF）
│   └── bin_to_edge.py
├── scripts/
│   ├── cross_compile.sh       # 交叉编译脚本
│   └── S99waferdirect         # Buildroot 开机自启
└── model/
    └── wafer_pc_int8.rknn     # INT8 量化模型（Git LFS）
```

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
qmake wafer_direct.pro
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

## 许可证

MIT License
