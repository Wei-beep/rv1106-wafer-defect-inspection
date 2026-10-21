#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一转换工具: 任意输入 → RV1106 直解 BIN (12352B)
支持格式:
  .bin  - 64x64 uint8 原始晶圆图 (0=背景, 1=良品, 2=失效), 4096B
  .png  - 彩色晶圆图 (export_png.py 生成: 背景深灰/良品绿/失效红)
  .stdf - 标准 SEMI STDF v4 (export_stdf.py 生成, DRR 记录逐 die 结果)

输出: 64x64x3 uint8 RGB (NHWC, 12288B) + 16×float32 特征 (64B) = 12352B
特征提取与 Qt waferfeature.cpp 完全一致, 含 Z-Score 归一化

用法:
  单文件:  python any_to_edge.py 000044.bin 000044_edge.bin
           python any_to_edge.py 007316.png 007316_edge.bin
           python any_to_edge.py 001592.stdf 001592_edge.bin
  批处理:  python any_to_edge.py --batch input_dir output_dir
"""
import numpy as np
import struct
import sys
import os
import glob

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── 归一化参数 (与 Qt mainwindow.cpp 完全一致) ──
FEAT_MEAN = np.array([0.5, 0.5, 0.2, 0.05, 0.02,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0.3, 1.0, 0.3], dtype=np.float32)
FEAT_STD  = np.array([0.1, 0.1, 0.1, 0.05, 0.01,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.3, 0.1], dtype=np.float32)


# ═══════════════════════════════════════════════════════════
#  特征提取 (与 Qt waferfeature.cpp 逐行对齐)
# ═══════════════════════════════════════════════════════════
def extract_features(wafer):
    """wafer: 64x64 uint8, 0=背景 1=良品 2=失效. 返回归一化 16 维特征."""
    h, w = wafer.shape
    feat = np.zeros(16, dtype=np.float32)

    defect_y, defect_x = np.where(wafer == 2)
    total_die = int(np.sum(wafer > 0))
    n_defect = len(defect_x)
    if n_defect == 0 or total_die == 0:
        return feat

    # [0][1] 重心
    cx = defect_x.mean() / w
    cy = defect_y.mean() / h
    feat[0] = cx
    feat[1] = cy

    # [2] 中心距离
    feat[2] = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    # [3] 失效密度
    feat[3] = n_defect / total_die

    # [4] 平均最近邻距离 (采样最多 100)
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

    # [13] 外接矩形紧凑度
    xmin, xmax = defect_x.min(), defect_x.max()
    ymin, ymax = defect_y.min(), defect_y.max()
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    bbox_area = bw * bh
    feat[13] = n_defect / bbox_area

    # [14] 长宽比
    feat[14] = bh / bw if bw > 0 else 1.0

    # [15] 主轴归一化长度
    diag = np.sqrt(bh ** 2 + bw ** 2)
    wafer_diag = np.sqrt(h ** 2 + w ** 2)
    feat[15] = diag / wafer_diag if wafer_diag > 0 else 0.0

    # Z-Score 归一化
    feat = (feat - FEAT_MEAN) / FEAT_STD
    return feat


# ═══════════════════════════════════════════════════════════
#  晶圆图 → 直解 BIN (公共输出)
# ═══════════════════════════════════════════════════════════
def wafer_to_edge_bin(wafer, output_path):
    """wafer: 64x64 uint8 (0/1/2). 写出 12352B edge bin."""
    if wafer.shape != (64, 64):
        if HAS_CV2:
            wafer = cv2.resize(wafer, (64, 64), interpolation=cv2.INTER_NEAREST)
        else:
            # 无 OpenCV 时用 numpy 最近邻缩放
            h, w = wafer.shape
            ys = (np.arange(64) * h / 64).astype(int)
            xs = (np.arange(64) * w / 64).astype(int)
            wafer = wafer[np.ix_(ys, xs)]

    # 图像: 0→0, 1→128, 2→255, 重复 3 通道, NHWC
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[wafer == 1] = 128
    img[wafer == 2] = 255

    # 特征
    feat = extract_features(wafer)

    with open(output_path, 'wb') as f:
        img.tofile(f)
        feat.tofile(f)
    return True


# ═══════════════════════════════════════════════════════════
#  格式 1: 原始 BIN (4096B, 64x64 uint8)
# ═══════════════════════════════════════════════════════════
def parse_raw_bin(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 4096:
        return data.reshape(64, 64)
    elif data.size == 12352:
        # 已经是 edge bin, 直接提取前 12288B 的灰度
        img = data[:12288].reshape(64, 64, 3)
        wafer = np.zeros((64, 64), dtype=np.uint8)
        wafer[img[:, :, 0] == 128] = 1
        wafer[img[:, :, 0] == 255] = 2
        return wafer
    else:
        raise ValueError(f"BIN 大小 {data.size}B 不支持 (需要 4096 或 12352)")


# ═══════════════════════════════════════════════════════════
#  格式 2: PNG (彩色晶圆图)
# ═══════════════════════════════════════════════════════════
def parse_png(path):
    if not HAS_CV2:
        raise RuntimeError("解析 PNG 需要 OpenCV: pip install opencv-python")
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError(f"无法读取 PNG: {path}")
    h, w = img.shape[:2]

    # 反向颜色映射 (export_png.py 的 COLOR_MAP, BGR 顺序):
    #   背景 (30,30,30)   → 0
    #   良品 (0,200,0)     → 1  (G 通道高)
    #   失效 (0,0,255)     → 2  (B 通道高, OpenCV BGR)
    wafer = np.zeros((h, w), dtype=np.uint8)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    wafer[(b > 128)] = 2            # 失效: 蓝色通道高
    wafer[(g > 128) & (b <= 128)] = 1  # 良品: 绿色通道高
    # 其余为背景(0)
    return wafer


# ═══════════════════════════════════════════════════════════
#  格式 3: STDF (SEMI v4, DRR 逐 die)
# ═══════════════════════════════════════════════════════════
def parse_stdf(path):
    """解析 export_stdf.py 生成的 STDF, 重建 64x64 晶圆图."""
    with open(path, 'rb') as f:
        data = f.read()

    pos = 0
    wafer_w, wafer_h = 64, 64
    wafer = None
    die_count = 0

    while pos < len(data):
        if pos + 4 > len(data):
            break
        rec_len, rec_type, rec_sub = struct.unpack_from('<HBB', data, pos)
        pos += 4
        rec_data = data[pos:pos + rec_len]
        pos += rec_len

        # WIR (2,10): 晶圆信息, 含宽高
        if rec_type == 2 and rec_sub == 10:
            # 格式: <BBIB (7B) + wafer_id(cstr) + defect_type(cstr) + <HH (w,h) + <hh
            rp = 7  # skip head, part type, node name
            # wafer_id
            if rp < len(rec_data):
                slen = rec_data[rp]
                rp += 1 + slen
            # defect_type
            if rp < len(rec_data):
                slen = rec_data[rp]
                rp += 1 + slen
            # width, height
            if rp + 4 <= len(rec_data):
                wafer_w, wafer_h = struct.unpack_from('<HH', rec_data, rp)
            wafer = np.zeros((wafer_h, wafer_w), dtype=np.uint8)

        # DRR (2,30): 逐 die 结果
        elif rec_type == 2 and rec_sub == 30:
            if wafer is None:
                wafer = np.zeros((wafer_h, wafer_w), dtype=np.uint8)
            # 格式: <BBIBhhB = 13B: head, part, x(int16), y(int16), test_flag
            if len(rec_data) >= 13:
                x, y, test_flag = struct.unpack_from('<hhB', rec_data, 7)
                if 0 <= x < wafer_w and 0 <= y < wafer_h:
                    if test_flag == 1:
                        wafer[y, x] = 1   # Pass → 良品
                    elif test_flag == 3:
                        wafer[y, x] = 2   # Fail → 失效
                    # test_flag==0 → 背景(默认0)
                    die_count += 1

    if wafer is None:
        raise ValueError(f"STDF 中未找到晶圆数据: {path}")
    return wafer


# ═══════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════
PARSERS = {
    '.bin': parse_raw_bin,
    '.png': parse_png,
    '.stdf': parse_stdf,
}


def convert_one(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in PARSERS:
        print(f"[跳过] 不支持的格式: {input_path}")
        return False
    try:
        wafer = PARSERS[ext](input_path)
        wafer_to_edge_bin(wafer, output_path)
        sz = os.path.getsize(output_path)
        print(f"[OK] {os.path.basename(input_path):20s} ({ext:5s}) → {os.path.basename(output_path)} ({sz}B)")
        return True
    except Exception as e:
        print(f"[失败] {input_path}: {e}")
        return False


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--batch':
        in_dir = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else 'edge_bin'
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for ext in PARSERS:
            files.extend(glob.glob(os.path.join(in_dir, '*' + ext)))
        files.sort()
        ok = 0
        for fp in files:
            name = os.path.splitext(os.path.basename(fp))[0]
            out = os.path.join(out_dir, name + '_edge.bin')
            if convert_one(fp, out):
                ok += 1
        print(f"\n批处理完成: {ok}/{len(files)} 个文件 → {out_dir}/")
    elif len(sys.argv) == 3:
        convert_one(sys.argv[1], sys.argv[2])
    else:
        print("用法:")
        print("  单文件: python any_to_edge.py input.(bin|png|stdf) output_edge.bin")
        print("  批处理: python any_to_edge.py --batch input_dir output_dir")
        print("\n支持格式: .bin (4096B原始) / .png (彩色) / .stdf (SEMI v4)")
        sys.exit(1)


if __name__ == '__main__':
    main()
#（注：内容由AI生成）
