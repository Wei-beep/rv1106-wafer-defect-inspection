# 踩坑记录 / Pitfalls & Lessons Learned

开发过程中遇到的关键问题、根因分析与解决方案。这些是项目工程深度的体现。

## 模型部署类

### 坑1：图像分支手动 int8 量化导致精度暴跌

- **现象**：板端推理准确率从 92% 降到 30% 以下
- **根因**：RKNN 的 image 输入已在量化模型中融合了归一化（scale=0.003922, zp=-128），板端必须喂 uint8 原始 RGB 像素 0~255。手动再做一次 int8 量化等于双重量化
- **解决**：image 分支喂原始 uint8，features 分支才需要手动 int8 量化 `q=round(x/0.099453+13)`

### 坑2：rknn_inputs_set 轻量库恒返回 -5

- **现象**：使用 rknn_inputs_set 轻量 API 始终返回错误码 -5
- **根因**：轻量库对双输入模型支持有 bug
- **解决**：使用标准 `rknn_input` 结构体 + `rknn_inputs_set` 完整 API 流程

### 坑3：零拷贝输出改 float32 被 NPU 拒绝

- **现象**：尝试将输出张量类型改为 float32 以跳过反量化，NPU 报错
- **根因**：量化模型的输出类型在转换时已固定为 int8，运行时不可更改
- **解决**：获取 int8 输出后手动反量化 `float=(int8-zp)*scale`

## 交叉编译类

### 坑4：glibc 交叉编译器 ABI 段错误

- **现象**：使用 arm-linux-gnueabihf-gcc（glibc）编译的程序在板上运行段错误
- **根因**：RV1106 Buildroot 使用 uclibc，glibc 编译的程序 ABI 不兼容
- **解决**：必须使用 `arm-rockchip830-linux-uclibcgnueabihf-gcc` 和 armhf-uclibc 版本的 librknnmrt

### 坑5：wafer_direct 权限不足

- **现象**：`./wafer_direct` 报 Permission denied，TCP 8080 连接被拒
- **根因**：交叉编译后的二进制没有执行权限，或开机自启脚本没加 chmod
- **解决**：`chmod +x wafer_direct`，并在 S99waferdirect 脚本中加入 chmod

## 性能与时序类

### 坑6：E2E 端到端时延 773ms（实际 2ms）

- **现象**：网页显示平均端到端时延 773ms，首帧时延 37818ms
- **根因**：E2E 计时起点设在 `recv()` 调用之前，包含了 TCP 等待数据到达的空闲时间
- **解决**：计时起点改为全量收包完成后，E2E 从 773ms 降到 ~2-3ms

## Qt 界面类

### 坑7：模式切换图层重叠

- **现象**：从产线模式切回研发模式时，产线面板覆盖在研发界面上
- **根因**：QWidget hide() 后仍保留 geometry 和 z-order
- **解决**：隐藏时 `setGeometry(0,0,0,0)` 清空占位 + `lower()` 置底，显示时 `raise()` 置顶 + `update()` 强制重绘

### 坑8：BusyBox netstat 不支持 -p 参数

- **现象**：`netstat -tlnp` 报 invalid option
- **根因**：Buildroot BusyBox 精简版 netstat 无 -p（进程名）选项
- **解决**：使用 `netstat -tln` 查看端口监听状态
