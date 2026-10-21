# 编译说明 / Build Guide

## 板端 C 服务交叉编译

### 环境要求

- x86 Ubuntu 主机
- 交叉编译器：`arm-rockchip830-linux-uclibcgnueabihf-gcc`
- RKNN 运行时库：librknnmrt（armhf-uclibc 版本）

### 编译命令

```bash
arm-rockchip830-linux-uclibcgnueabihf-gcc board/wafer_direct.c -o wafer_direct \
  -I./ \
  -L$HOME/.local/lib/python3.8/site-packages/rknn/3rdparty/\
platform-tools/runtime/Linux/librknn_api/armhf-uclibc \
  -lrknnmrt -lm -lpthread -O2
```

或使用脚本：

```bash
chmod +x scripts/cross_compile.sh
./scripts/cross_compile.sh
```

### 注意事项

- **必须使用 uclibc 版本**的交叉编译器和 librknnmrt，glibc 版本会因 ABI 不兼容导致段错误
- `-O2` 优化开启，板端性能敏感
- 链接 `-lm`（数学库）和 `-lpthread`（多线程）

## Qt 上位机编译

### 环境要求

- Qt 5.x（Windows / Linux）
- MinGW 或 MSVC 编译器（Windows）/ GCC（Linux）

### 编译步骤

```bash
cd host
qmake WaferInspector.pro
make -j4
```

### 运行依赖

- Qt 运行时库（Qt5Core, Qt5Gui, Qt5Widgets, Qt5Network）
- Windows 部署时使用 `windeployqt` 拷贝依赖 DLL

## 模型训练

详见 [train/README 说明](../README.md#模型训练) 及 [training_results.md](training_results.md)。
