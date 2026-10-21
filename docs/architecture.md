# 系统架构与协议说明 / Architecture & Protocol

## 整体架构

系统采用"训练-转换-部署-监控"四层架构：

```
┌─────────────────────────────────────────────────────────────┐
│  训练层  PyTorch 双输入CNN训练 → ONNX导出                     │
├─────────────────────────────────────────────────────────────┤
│  转换层  RKNN Toolkit INT8量化 → wafer_pc_int8.rknn          │
├─────────────────────────────────────────────────────────────┤
│  部署层  RV1106 板端 C 服务 (wafer_direct)                    │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │direct    │tcp_loop  │http_loop │sse_beat  │udp_bcast │  │
│  │目录监控  │TCP 8080  │HTTP 8081 │SSE心跳   │UDP 8090  │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
│                    RKNN NPU 推理 (2.12ms/帧)                 │
├─────────────────────────────────────────────────────────────┤
│  应用层  Qt上位机(C++)  +  SSE网页看板(HTML/JS)              │
└─────────────────────────────────────────────────────────────┘
```

## 端口分配

| 端口 | 协议 | 功能 |
|---|---|---|
| 8080 | TCP | 研发模式推理接口（RFNS/RFNR 二进制协议） |
| 8081 | HTTP | 网页看板 + /api/stream SSE + /api/upload 文件上传 |
| 8090 | UDP | 板子 IP 广播发现（每 2 秒） |

## 数据流

### 产线直解模式
```
产线 STDF/BIN/PNG
    → Qt WaferConverter 转换为 12352B edge bin
    → HTTP POST /api/upload 上传到板子 incoming/
    → 板端 direct_loop 每 300ms 扫描拾取
    → RKNN NPU 推理
    → SSE 推送结果到网页看板
    → 文件移到 processed/
```

### 研发调试模式
```
Qt 加载图像/特征
    → TCP 8080 发送 12356B (RFNS + img + feat)
    → 板端 NPU 推理
    → TCP 返回 44B (RFNR + pred + conf + 8×logits)
    → Qt 展示结果表格 + 热力图
```

## 统一 Edge Bin 格式

所有数据源（STDF/BIN/PNG）经转换后统一为 12352 字节：

```
偏移        长度        内容
0           12288       64×64×3 uint8 RGB (NHWC)
                          0→(0,0,0) 背景
                          1→(128,128,128) 良品
                          2→(255,255,255) 失效
12288       64          16×float32 特征 (Z-Score 归一化)
```

## TCP 推理协议（RFNS/RFNR）

### 请求（Qt → 板，12356 字节）
```
偏移  长度  内容
0     4     魔数 "RFNS" (0x52 0x46 0x4E 0x53)
4     12288 图像数据 (64×64×3 uint8 RGB NHWC)
12292 64    特征数据 (16×float32)
```

### 响应（板 → Qt，44 字节）
```
偏移  长度  内容
0     4     魔数 "RFNR" (0x52 0x46 0x4E 0x52)
4     4     pred (int32, 预测类别索引 0-7)
8     4     conf (float32, 最高置信度)
12    32    8×logits (float32, 各类别原始输出)
```

### 收包注意事项
- TCP 是流协议，单次 `recv()` 可能收不全，必须用 `recv_n()` 循环收满指定字节数
- 同理发送用 `send_n()` 确保全量发出
- 这是避免粘包/半包的标准做法

## RKNN 量化参数

| 张量 | 类型 | scale | zero_point | 说明 |
|---|---|---|---|---|
| image 输入 | uint8 | 0.003922 | -128 | NPU 内部融合归一化，板端喂原始像素 |
| features 输入 | int8 | 0.099453 | 13 | 手动量化 q=round(x/0.099453+13) |
| output 输出 | int8 | 0.255337 | 3 | 反量化 float=(int8-zp)*scale |

## 缺陷类别映射

| 索引 | 类别 | 说明 |
|---|---|---|
| 0 | Center | 中心聚团 |
| 1 | Donut | 环形 |
| 2 | Edge-Loc | 边缘局部 |
| 3 | Edge-Ring | 边缘环 |
| 4 | Loc | 局部聚团 |
| 5 | Near-full | 大片聚集 |
| 6 | Random | 随机散点 |
| 7 | Scratch | 划痕 |

## HTTP API

| 接口 | 方法 | 功能 |
|---|---|---|
| `/` | GET | 网页看板 HTML |
| `/api/stream` | GET | SSE 实时事件流 |
| `/api/upload?name=xxx` | POST | 上传 edge bin 到 incoming/ |
| `/api/device/status` | GET | 设备状态 JSON |

## SSE 事件格式

```
data: {"type":"infer","file":"xxx.bin","pred":"Edge-Ring","conf":0.955,"second":"Loc","npu_ms":2.14,"source":"direct"}

data: {"type":"status","uptime":"01:23:45","total":150,"processed":150,"fps":45.2}
```

## 为什么选 SSE 而不是 WebSocket

| 特性 | SSE (Server-Sent Events) | WebSocket |
|---|---|---|
| 通信方向 | 服务端→客户端单向 | 双向 |
| 协议 | HTTP 长连接 | 独立 WS 协议（需握手升级） |
| 板端实现复杂度 | 极低（写 HTTP 头后循环 write） | 高（握手/帧解析/掩码/心跳） |
| 自动重连 | 浏览器原生支持 | 需手动实现 |
| 适用场景 | 监控数据推送（本项目） | 双向交互/聊天室 |

板端仅 64MB 内存，SSE 只需在 HTTP 响应头写入 `Content-Type: text/event-stream` 然后循环 write 数据，实现成本远低于 WebSocket。看板是单向推送（板→浏览器），SSE 完全满足需求。
