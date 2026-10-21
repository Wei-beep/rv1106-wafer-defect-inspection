#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WM-811K 原始 BIN → RV1106 直解 BIN 转换工具
输入: 64x64 uint8 单通道 (0=背景, 1=良品, 2=失效), 4096字节
输出: 64x64x3 uint8 RGB (NHWC) + 16×float32 特征, 12352字节
      图像映射: 0→(0,0,0), 1→(128,128,128), 2→(255,255,255)
      特征提取与 Qt waferfeature.cpp 完全一致, 含 Z-Score 归一化

用法:
  单文件:  python bin_to_edge.py 000044.bin 000044_edge.bin
  批处理:  python bin_to_edge.py --batch input_dir output_dir
"""
import numpy as np
import sys
import os
import glob

# 与 Qt mainwindow.cpp FEAT_MEAN/FEAT_STD 完全一致
FEAT_MEAN = np.array([0.5, 0.5, 0.2, 0.05, 0.02,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0.3, 1.0, 0.3], dtype=np.float32)
FEAT_STD  = np.array([0.1, 0.1, 0.1, 0.05, 0.01,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.3, 0.1], dtype=np.float32)


def extract_features(wafer):
    """wafer: 64x64 uint8, 0=背景 1=良品 2=失效. 返回归一化后的 16 维特征."""
    h, w = wafer.shape
    feat = np.zeros(16, dtype=np.float32)

    defect_y, defect_x = np.where(wafer == 2)
    total_die = int(np.sum(wafer > 0))
    n_defect = len(defect_x)
    if n_defect == 0 or total_die == 0:
        return feat

    # [0][1] 重心坐标 (归一化到 [0,1])
    cx = defect_x.mean() / w
    cy = defect_y.mean() / h
    feat[0] = cx
    feat[1] = cy

    # [2] 中心距离
    feat[2] = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    # [3] 失效密度
    feat[3] = n_defect / total_die

    # [4] 平均最近邻距离 (采样最多 100 点)
    n_sample = min(n_defect, 100)
    if n_defect > n_sample:
        idx = np.random.choice(n_defect, n_sample, replace=False)
    else:
        idx = np.arange(n_defect)
    sx = defect_x[idx].astype(np.float64)
    sy = defect_y[idx].astype(np.float64)
    sum_nn = 0.0
    for i in range(n_sample):
        dists = np.sqrt((sx[i] - sx) ** 2 + (sy[i] - sy) ** 2)
        dists[i] = 1e9
        sum_nn += dists.min()
    feat[4] = sum_nn / n_sample / max(h, w)

    # [5]-[12] 8 环带分布
    ccx, ccy = w / 2.0, h / 2.0
    max_r = np.sqrt(ccx ** 2 + ccy ** 2)
    r = np.sqrt((defect_x - ccx) ** 2 + (defect_y - ccy) ** 2)
    hist = np.zeros(8, dtype=np.float32)
    for ri in r:
        b = int(ri / max_r * 8)
        if b >= 8:
            b = 7
        hist[b] += 1
    feat[5:13] = hist / n_defect

    # [13] 外接矩形紧凑度 (缺陷数/外接矩形面积)
    xmin, xmax = defect_x.min(), defect_x.max()
    ymin, ymax = defect_y.min(), defect_y.max()
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    bbox_area = bw * bh
    feat[13] = n_defect / bbox_area

    # [14] 长宽比 (高/宽)
    feat[14] = bh / bw if bw > 0 else 1.0

    # [15] 主轴归一化长度 (外接矩形对角线/晶圆对角线)
    diag = np.sqrt(bh ** 2 + bw ** 2)
    wafer_diag = np.sqrt(h ** 2 + w ** 2)
    feat[15] = diag / wafer_diag if wafer_diag > 0 else 0.0

    # Z-Score 归一化 (与 Qt 一致)
    feat = (feat - FEAT_MEAN) / FEAT_STD
    return feat


def convert_one(input_path, output_path):
    """转换单个 bin 文件."""
    data = np.fromfile(input_path, dtype=np.uint8)
    if data.size != 4096:
        print(f"[跳过] {input_path}: 大小 {data.size}B ≠ 4096B")
        return False
    wafer = data.reshape(64, 64)

    # 图像: 0→0, 1→128, 2→255, 重复 3 通道, NHWC
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[wafer == 1] = 128
    img[wafer == 2] = 255

    # 特征
    feat = extract_features(wafer)

    # 写出: 图像 12288B + 特征 64B = 12352B
    with open(output_path, 'wb') as f:
        img.tofile(f)
        feat.tofile(f)
    return True


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--batch':
        in_dir = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else 'edge_bin'
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(in_dir, '*.bin')))
        ok = 0
        for fp in files:
            name = os.path.basename(fp)
            out = os.path.join(out_dir, name.replace('.bin', '_edge.bin'))
            if convert_one(fp, out):
                ok += 1
        print(f"批处理完成: {ok}/{len(files)} 个文件 → {out_dir}/")
    elif len(sys.argv) == 3:
        if convert_one(sys.argv[1], sys.argv[2]):
            sz = os.path.getsize(sys.argv[2])
            print(f"已生成 {sys.argv[2]} ({sz}B)")
    else:
        print("用法:")
        print("  单文件: python bin_to_edge.py input.bin output.bin")
        print("  批处理: python bin_to_edge.py --batch input_dir output_dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
#（注：内容由AI生成）
