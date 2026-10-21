# 板端部署说明 / Deployment Guide

## 硬件要求

- 瑞芯微 RV1106 开发板（Luckfox Pico 或同类）
- 64MB DDR2 内存
- Buildroot Linux（uclibc, BusyBox）
- 网络连接（有线/WiFi）

## 部署步骤

### 1. 拷贝文件到板子

```bash
scp wafer_direct root@<板子IP>:/root/wafer_test/8_class/
scp model/wafer_pc_int8.rknn root@<板子IP>:/root/wafer_test/8_class/
```

### 2. 板子上运行

```bash
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
[INFO] TCP inference on :8080 (edge NPU mode)
[INFO] Web(SSE) dashboard on :8081
[INFO] direct mode: watching ./incoming/ (BIN 12352 bytes), output -> ./processed/
```

### 3. 验证

```bash
# 查看端口监听（BusyBox 精简版 netstat 不带 -p）
netstat -tln | grep -E '8080|8081'

# 浏览器打开网页看板
# http://<板子IP>:8081

# 放一个 edge bin 到 incoming/ 测试直解
cp test.bin incoming/
# 几秒后应在网页看板看到推理结果，文件被移到 processed/
```

## 开机自启

将 `scripts/S99waferdirect` 拷贝到板子的 `/etc/init.d/`：

```bash
scp scripts/S99waferdirect root@<板子IP>:/etc/init.d/
ssh root@<板子IP> "chmod +x /etc/init.d/S99waferdirect"
```

Buildroot 会按 `S##` 编号顺序在开机时自动执行 `start`。重启后 `wafer_direct` 自动运行，产线无需 SSH 操作。

### 自启脚本内容

```sh
#!/bin/sh
case "$1" in
    start)
        cd /root/wafer_test/8_class
        chmod +x wafer_direct
        export LD_LIBRARY_PATH=/oem/usr/lib:./
        mkdir -p incoming processed
        ./wafer_direct wafer_pc_int8.rknn &
        ;;
    stop)
        killall wafer_direct
        ;;
esac
```

## 运行时库路径

RKNN 运行时库 `librknnmrt.so` 位于板子的 `/oem/usr/lib/`，必须设置：

```bash
export LD_LIBRARY_PATH=/oem/usr/lib:./
```

否则会报 `error while loading shared libraries: librknnmrt.so: cannot open shared object file`。

## 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| Permission denied | 二进制无执行权限 | `chmod +x wafer_direct` |
| librknnmrt.so not found | 未设 LD_LIBRARY_PATH | `export LD_LIBRARY_PATH=/oem/usr/lib:./` |
| TCP 8080 connection refused | 服务未启动或崩溃 | 检查开机自启，手动运行看报错 |
| netstat -p 报错 | BusyBox 精简版不支持 -p | 用 `netstat -tln` |
| 网页看板打不开 | 防火墙或端口未监听 | 检查 8081 端口，确认服务在跑 |
