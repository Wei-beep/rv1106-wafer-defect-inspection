#!/bin/bash
# RV1106 wafer_direct 交叉编译脚本
# 用法: ./cross_compile.sh
# 依赖: arm-rockchip830-linux-uclibcgnueabihf-gcc 在 PATH 中

set -e

CROSS=arm-rockchip830-linux-uclibcgnueabihf-gcc
SRC=board/wafer_direct.c
OUT=wafer_direct

# RKNN 运行时库路径（根据实际安装位置调整）
RKNN_LIB=$HOME/.local/lib/python3.8/site-packages/rknn/3rdparty/platform-tools/runtime/Linux/librknn_api/armhf-uclibc

echo "[*] Cross-compiling $SRC -> $OUT ..."
$CROSS $SRC -o $OUT \
  -I./ \
  -L$RKNN_LIB \
  -lrknnmrt -lm -lpthread -O2

echo "[+] Done: $OUT"
file $OUT
ls -lh $OUT
