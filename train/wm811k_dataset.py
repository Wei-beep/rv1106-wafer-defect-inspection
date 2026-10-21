#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WM-811K Wafer Map Data Preprocessing & Model - Python (PyTorch)
===============================================================
WM-811K 晶圆图数据预处理与模型定义（PyTorch版）

核心流程 / Core Pipeline:
    加载PKL -> 修复标签格式 -> 重建晶圆图 -> resize/padding归一化 -> 三通道编码
    -> 约束数据增强 -> 手工空间特征提取(16维) -> 特征标准化
    -> 双分支CNN(图像分支+手工特征分支 + SE注意力) -> Focal Loss训练 -> 导出ONNX -> 推理对齐校验

修复记录 / Changelog:
    [FIX 1] failureType 嵌套列表格式修复
    [FIX 2] 手工特征增加标准化（仅训练集统计mean/std）
    [FIX 3] FocalLoss 支持传入类别权重 alpha
    [FIX 4] ONNX导出开启 do_constant_folding 常量折叠优化
    [FIX 5] ONNX导出后自动校验 PyTorch 与 ONNX Runtime 输出一致性
    [FIX 6] Dataset 支持传入 feature_mean/std 做标准化
    [FIX 7] 增加数据可视化调试函数
    [FIX 8] 增加 pandas 旧版本pickle兼容层
    [FIX 9] pickle.load 增加 encoding='latin1'
    [FIX 10] LSWMD.pkl DataFrame 自动转 records
    [FIX 11] 扩充约束型数据增强：平移、缩放、随机die缺失
    [OPT 1]  新增SE通道注意力模块，提升特征提取效率
    [OPT 2]  手工特征新增紧凑度、长宽比、主轴长度，提升Loc/Center/Scratch区分度
    [OPT 3]  轻微手动类别加权，定向提升短板类性能
    [OPT 4]  通道加宽版：卷积通道32/64/128 → 64/128/256，提升特征容量
"""

import pickle
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

# ──────────────────────────────────────
# 1. 全局配置 / Global Configuration
# ──────────────────────────────────────

# 8类缺陷模式（已去掉none，聚焦缺陷模式识别）
DEFECT_CLASSES = [
    'Center',       # 中心聚团 / Center cluster
    'Donut',        # 环形 / Donut-shaped
    'Edge-Loc',     # 边缘局部 / Edge-localized
    'Edge-Ring',    # 边缘环 / Edge ring
    'Loc',          # 局部聚团 / Localized cluster
    'Random',       # 随机散点 / Random scatter
    'Scratch',      # 划痕 / Scratch
    'Near-full',    # 大片聚集 / Near-full coverage
]

CLASS_TO_IDX = {c: i for i, c in enumerate(DEFECT_CLASSES)}
NUM_CLASSES = len(DEFECT_CLASSES)

TARGET_SIZE = 64      # CNN输入统一尺寸 / Uniform input size
FEATURE_DIM = 16     # 手工空间特征维度 / Handcrafted feature dimension
                     # 5全局 + 8环带 + 3形状(紧凑度/长宽比/主轴长度)

# ──────────────────────────────────────
# 2. pandas 旧版本兼容层
# ──────────────────────────────────────

def _patch_pandas_compat():
    """[FIX 8] pandas 旧版本pickle兼容层"""
    try:
        import pandas as pd
        import pandas.core.indexes as core_indexes

        if 'pandas.indexes' not in sys.modules:
            sys.modules['pandas.indexes'] = core_indexes

        submodules = ['base', 'numeric', 'range', 'multi', 'datetimes',
                       'timedeltas', 'period', 'interval', 'category']
        for sub in submodules:
            old_name = f'pandas.indexes.{sub}'
            if old_name not in sys.modules:
                try:
                    new_mod = getattr(core_indexes, sub)
                    sys.modules[old_name] = new_mod
                except AttributeError:
                    pass

        try:
            import pandas.core.tseries as core_tseries
            if 'pandas.tseries' not in sys.modules:
                sys.modules['pandas.tseries'] = core_tseries
        except ImportError:
            pass

        try:
            import pandas._libs as core_libs
            if 'pandas._libs' not in sys.modules:
                sys.modules['pandas._libs'] = core_libs
        except ImportError:
            pass

    except ImportError:
        pass

# ──────────────────────────────────────
# 3. 数据加载 / Data Loading
# ──────────────────────────────────────

def load_wm811k(pkl_path: str) -> list:
    """加载WM-811K数据集（PKL格式）"""
    _patch_pandas_compat()

    with open(pkl_path, 'rb') as f:
        try:
            data = pickle.load(f, encoding='latin1')
        except TypeError:
            data = pickle.load(f)

    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            print(f"[Data] 检测到DataFrame格式 / Detected DataFrame format "
                   f"({len(data)} rows, columns: {list(data.columns)})")
            data = data.to_dict('records')
    except ImportError:
        pass

    labeled = []
    for d in data:
        if not isinstance(d, dict):
            continue

        ft = d.get('failureType')

        if isinstance(ft, (list, np.ndarray)):
            if len(ft) > 0:
                ft = ft[0]
                if isinstance(ft, (list, np.ndarray)):
                    ft = ft[0] if len(ft) > 0 else None
            else:
                ft = None

        if ft is not None:
            ft = str(ft).strip()

        if ft in CLASS_TO_IDX:
            sample = dict(d)
            sample['failureType'] = ft
            labeled.append(sample)

    print(f"[Data] 已加载 / Loaded {len(labeled)} 张带标签晶圆图 / labeled wafer maps "
           f"(共 / total {len(data)} 张)")
    return labeled


def stratified_split(samples: list, train_ratio: float = 0.7,
                      val_ratio: float = 0.15, seed: int = 42):
    """分层划分训练/验证/测试集"""
    np.random.seed(seed)

    by_class = {}
    for s in samples:
        c = s['failureType']
        by_class.setdefault(c, []).append(s)

    train, val, test = [], [], []
    for c, items in by_class.items():
        np.random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    print(f"[Split] 训练集/train={len(train)}, 验证集/val={len(val)}, "
           f"测试集/test={len(test)}")
    return train, val, test

# ──────────────────────────────────────
# 4. 晶圆图预处理 / Wafer Map Preprocessing
# ──────────────────────────────────────

def resize_wafer_map(wafer_map: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """统一resize/padding，INTER_NEAREST保持三值离散性"""
    import cv2
    h, w = wafer_map.shape

    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(wafer_map.astype(np.uint8), (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST)

    padded = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    padded[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return padded


def encode_three_channel(wafer_map: np.ndarray) -> np.ndarray:
    """单通道编码为3通道浮点张量"""
    h, w = wafer_map.shape
    encoded = np.zeros((3, h, w), dtype=np.float32)
    encoded[0] = (wafer_map == 2).astype(np.float32)  # 失效
    encoded[1] = (wafer_map == 1).astype(np.float32)  # 良品
    encoded[2] = (wafer_map == 0).astype(np.float32)  # 背景
    return encoded

# ──────────────────────────────────────
# 5. 手工空间特征提取
# ──────────────────────────────────────

def extract_spatial_features(wafer_map: np.ndarray) -> np.ndarray:
    """
    提取16维手工空间特征
    新增紧凑度、长宽比、主轴长度，提升Loc/Center/Scratch区分能力
    """
    h, w = wafer_map.shape
    defect_mask = (wafer_map == 2)
    total_die = np.sum(wafer_map > 0)
    n_defect = np.sum(defect_mask)

    if n_defect == 0 or total_die == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    ys, xs = np.where(defect_mask)

    # [0-1] 重心坐标（归一化）
    cx, cy = xs.mean() / w, ys.mean() / h
    # [2] 偏离中心距离
    dist_from_center = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    # [3] 失效密度
    density = n_defect / total_die

    # [5-12] 环带分布（8个径向区间）
    center_y, center_x = h / 2.0, w / 2.0
    radii = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    max_r = np.sqrt(center_x ** 2 + center_y ** 2)
    hist, _ = np.histogram(radii, bins=8, range=(0, max_r), density=True)

    # [4] 随机度：平均最近邻距离
    if n_defect > 100:
        idx = np.random.choice(n_defect, 100, replace=False)
        xs_s, ys_s = xs[idx], ys[idx]
    else:
        xs_s, ys_s = xs, ys

    if len(xs_s) > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([xs_s, ys_s]))
        dists, _ = tree.query(np.column_stack([xs_s, ys_s]), k=2)
        avg_nn = dists[:, 1].mean() / max(h, w)
    else:
        avg_nn = 0.0

    # 形状特征：区分Loc/Center/Scratch
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    bbox_area = bbox_h * bbox_w

    # [13] 紧凑度：缺陷面积 / 外接矩形面积
    compactness = n_defect / bbox_area

    # [14] 长宽比：外接矩形高/宽
    aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 1.0

    # [15] 缺陷主轴长度归一化：外接矩形对角线 / 晶圆直径
    bbox_diag = np.sqrt(bbox_h ** 2 + bbox_w ** 2)
    wafer_diag = np.sqrt(h ** 2 + w ** 2)
    major_axis_norm = bbox_diag / wafer_diag if wafer_diag > 0 else 0.0

    # 组装16维特征
    features = np.concatenate([
        [cx, cy, dist_from_center, density, avg_nn],  # 5维全局
        hist,                                          # 8维环带
        [compactness, aspect_ratio, major_axis_norm]   # 3维形状
    ]).astype(np.float32)

    return features[:FEATURE_DIM]


def compute_feature_stats(train_samples: list, save_path: str = "feature_norm.npz"):
    """仅在训练集上统计手工特征均值/标准差"""
    print(f"[Feature Stats] 正在统计训练集手工特征均值/标准差（{len(train_samples)} 样本）...")
    feat_list = []
    for i, s in enumerate(train_samples):
        wm = resize_wafer_map(s['waferMap'], TARGET_SIZE)
        feat = extract_spatial_features(wm)
        feat_list.append(feat)
        if (i + 1) % 20000 == 0:
            print(f"  已处理 / Processed {i + 1}/{len(train_samples)}")

    feat_arr = np.stack(feat_list, axis=0)
    mean = np.mean(feat_arr, axis=0).astype(np.float32)
    std = np.std(feat_arr, axis=0).astype(np.float32)

    std[std < 1e-6] = 1.0

    np.savez(save_path, mean=mean, std=std)
    print(f"[Feature Stats] 已保存 / Saved to {save_path}")
    return mean, std


def load_feature_stats(path: str = "feature_norm.npz"):
    """加载手工特征标准化参数"""
    data = np.load(path)
    return data['mean'], data['std']


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score标准化"""
    return (features - mean) / std

# ──────────────────────────────────────
# 6. PyTorch Dataset
# ──────────────────────────────────────

class WM811KDataset(Dataset):
    """WM-811K 晶圆图 PyTorch Dataset"""

    def __init__(self, samples: list, target_size: int = TARGET_SIZE,
                 augment: bool = False, use_features: bool = True,
                 feature_mean: np.ndarray = None, feature_std: np.ndarray = None):
        self.samples = samples
        self.target_size = target_size
        self.augment = augment
        self.use_features = use_features
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        wafer_map = sample['waferMap']
        label = CLASS_TO_IDX[sample['failureType']]

        # Step 1: Resize归一化
        wm = resize_wafer_map(wafer_map, self.target_size)

        # Step 2: 约束数据增强（仅训练集）
        if self.augment:
            import cv2

            # 90°倍数旋转
            k = np.random.randint(0, 4)
            wm = np.rot90(wm, k)

            # 水平翻转
            if np.random.random() > 0.5:
                wm = np.fliplr(wm)

            # 随机平移（-2~2 格）
            if np.random.random() > 0.5:
                shift_y = np.random.randint(-2, 3)
                shift_x = np.random.randint(-2, 3)
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                wm = cv2.warpAffine(wm, M, (self.target_size, self.target_size),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)

            # 随机缩放（0.95~1.05 倍）
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                new_size = int(self.target_size * scale)
                wm_scaled = cv2.resize(wm, (new_size, new_size),
                                        interpolation=cv2.INTER_NEAREST)
                if new_size > self.target_size:
                    start = (new_size - self.target_size) // 2
                    wm = wm_scaled[start:start + self.target_size,
                                  start:start + self.target_size]
                else:
                    pad = (self.target_size - new_size) // 2
                    wm = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
                    wm[pad:pad + new_size, pad:pad + new_size] = wm_scaled

            # 随机 die 缺失（1%~3%）
            if np.random.random() > 0.5:
                drop_ratio = np.random.uniform(0.01, 0.03)
                valid_mask = (wm > 0)
                valid_indices = np.argwhere(valid_mask)
                n_drop = int(len(valid_indices) * drop_ratio)
                if n_drop > 0:
                    drop_idx = np.random.choice(len(valid_indices), n_drop, replace=False)
                    for y, x in valid_indices[drop_idx]:
                        wm[y, x] = 0

            wm = np.ascontiguousarray(wm)

        # Step 3: 三通道编码
        image = encode_three_channel(wm)
        image = torch.from_numpy(image)

        # Step 4: 手工特征提取 + 标准化
        if self.use_features:
            features = extract_spatial_features(wm)
            if self.feature_mean is not None and self.feature_std is not None:
                features = normalize_features(features, self.feature_mean, self.feature_std)
            features = torch.from_numpy(features)
            return image, features, label

        return image, label

# ──────────────────────────────────────
# 7. 类别不均衡处理
# ──────────────────────────────────────

def get_class_weights(samples: list) -> np.ndarray:
    """
    计算类别权重（与样本数成反比）
    轻微手动加权：给Loc、Scratch两个短板类乘以1.25倍
    """
    counts = Counter(s['failureType'] for s in samples)
    total = len(samples)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c, idx in CLASS_TO_IDX.items():
        if c in counts and counts[c] > 0:
            weights[idx] = total / (NUM_CLASSES * counts[c])

    # 定向提升短板类
    boost_classes = ['Loc', 'Scratch']
    boost_factor = 1.25
    for c in boost_classes:
        idx = CLASS_TO_IDX[c]
        weights[idx] *= boost_factor

    return weights


def get_weighted_sampler(samples: list) -> WeightedRandomSampler:
    """创建WeightedRandomSampler"""
    class_weights = get_class_weights(samples)
    sample_weights = [class_weights[CLASS_TO_IDX[s['failureType']]]
                       for s in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(samples),
                                  replacement=True)


def get_dataloaders(pkl_path: str, batch_size: int = 64, num_workers: int = 4,
                     use_features: bool = True, use_weighted_sampler: bool = True,
                     feature_mean=None, feature_std=None):
    """创建训练/验证/测试 DataLoader"""
    data = load_wm811k(pkl_path)
    train_samples, val_samples, test_samples = stratified_split(data)

    train_ds = WM811KDataset(train_samples, augment=True,
                              use_features=use_features,
                              feature_mean=feature_mean, feature_std=feature_std)
    val_ds = WM811KDataset(val_samples, augment=False,
                            use_features=use_features,
                            feature_mean=feature_mean, feature_std=feature_std)
    test_ds = WM811KDataset(test_samples, augment=False,
                             use_features=use_features,
                             feature_mean=feature_mean, feature_std=feature_std)

    sampler = get_weighted_sampler(train_samples) if use_weighted_sampler else None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               sampler=sampler, num_workers=num_workers,
                               drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_samples, val_samples, test_samples

# ──────────────────────────────────────
# 8. Focal Loss
# ──────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss: 专门为类别不均衡设计的损失函数"""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ──────────────────────────────────────
# 9. SE 通道注意力模块
# ──────────────────────────────────────

class SELayer(nn.Module):
    """
    Squeeze-Excitation 通道注意力
    轻量级注意力，自动学习通道权重
    """
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ──────────────────────────────────────
# 10. 轻量级CNN模型（通道加宽版 + SE注意力 + 双分支融合）
# ──────────────────────────────────────

class WaferMapCNN(nn.Module):
    """
    [OPT 4] 通道加宽版：卷积通道32/64/128 → 64/128/256
    轻量级CNN，支持手工特征与深度特征融合
    """

    def __init__(self, num_classes: int = NUM_CLASSES, feature_dim: int = FEATURE_DIM,
                 use_fusion: bool = True):
        super().__init__()
        self.use_fusion = use_fusion
        self.feature_dim = feature_dim

        # 图像分支：3个卷积块 + SE注意力（通道加宽）
        self.features = nn.Sequential(
            # Block 1: 3 → 64, 64→32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SELayer(64, reduction=4),

            # Block 2: 64 → 128, 32→16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SELayer(128, reduction=4),

            # Block 3: 128 → 256, 16→8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SELayer(256, reduction=4),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = 256  # 通道加宽后输出维度

        # 手工特征投影分支
        if use_fusion:
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
            )
            fusion_dim = cnn_out_dim + 128  # 256 + 128 = 384
        else:
            self.feature_proj = None
            fusion_dim = cnn_out_dim  # 256

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        cnn_feat = self.features(x)
        cnn_feat = self.gap(cnn_feat).flatten(1)  # [B, 256]

        if self.use_fusion and features is not None:
            hand_feat = self.feature_proj(features)  # [B, 128]
            fused = torch.cat([cnn_feat, hand_feat], dim=1)  # [B, 384]
        else:
            fused = cnn_feat  # [B, 256]

        logits = self.classifier(fused)
        return logits

# ──────────────────────────────────────
# 11. ONNX导出 + 推理对齐校验
# ──────────────────────────────────────

def export_to_onnx(model: nn.Module, onnx_path: str = 'wafer_model.onnx',
                    use_features: bool = True, opset_version: int = 14):
    """导出ONNX并校验输出一致性"""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, TARGET_SIZE, TARGET_SIZE, device=device)
    dummy_features = torch.randn(1, FEATURE_DIM, device=device) if use_features else None

    print(f"[ONNX] 正在导出 / Exporting to {onnx_path} ...")

    if use_features:
        torch.onnx.export(
            model,
            (dummy_input, dummy_features),
            onnx_path,
            input_names=['image', 'features'],
            output_names=['logits'],
            dynamic_axes={
                'image': {0: 'batch'},
                'features': {0: 'batch'},
                'logits': {0: 'batch'},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
    else:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=opset_version,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[ONNX] 导出完成 / Exported: {onnx_path} ({size_mb:.2f} MB)")

    # 对齐校验
    print(f"[ONNX] 正在校验 PyTorch ↔ ONNX Runtime 输出一致性...")
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  输入 / Inputs: {input_names}")
    print(f"  输出 / Outputs: {output_names}")

    ort_inputs = {'image': dummy_input.cpu().numpy()}
    if use_features:
        ort_inputs['features'] = dummy_features.cpu().numpy()

    with torch.no_grad():
        if use_features:
            pt_out = model(dummy_input, dummy_features).cpu().numpy()
        else:
            pt_out = model(dummy_input).cpu().numpy()

    ort_out = sess.run(['logits'], ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    print(f"  最大误差 / Max difference:  {max_diff:.2e}")
    print(f"  平均误差 / Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-2:
        print(f"  ✅ 校验通过 / Alignment check PASSED (max_diff < 1e-2)")
    else:
        print(f"  ⚠️  误差偏大 / Alignment check WARNING (max_diff >= 1e-2)")

    return onnx_path

# ──────────────────────────────────────
# 12. 可视化调试
# ──────────────────────────────────────

def visualize_wafer_map(wafer_map: np.ndarray, title: str = "", save_path: str = None):
    """可视化晶圆图（调试用）"""
    import matplotlib.pyplot as plt

    h, w = wafer_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[wafer_map == 1] = [0, 200, 0]    # 绿色=良品
    rgb[wafer_map == 2] = [255, 0, 0]    # 红色=失效
    rgb[wafer_map == 0] = [30, 30, 30]   # 深灰=背景

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Viz] 已保存 / Saved to {save_path}")
    else:
        plt.show()

# ──────────────────────────────────────
# 13. 快速自测
# ──────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  WM-811K 数据模块自测 / Module Self-test")
    print("=" * 60)

    print("\n[Test 0] pandas兼容层 / Pandas compatibility patch")
    _patch_pandas_compat()
    import pandas as pd
    print(f"  pandas 版本 / pandas version: {pd.__version__}")
    print(f"  pandas.indexes 已注册: {'pandas.indexes' in sys.modules}")

    print("\n[Test 1] 模型前向传播 / Model forward pass")
    model = WaferMapCNN(use_fusion=True)
    x = torch.randn(2, 3, TARGET_SIZE, TARGET_SIZE)
    f = torch.randn(2, FEATURE_DIM)
    out = model(x, f)
    print(f"  输入 / Input: image={x.shape}, features={f.shape}")
    print(f"  输出 / Output: logits={out.shape}")
    print(f"  参数量 / Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model_no_fuse = WaferMapCNN(use_fusion=False)
    out2 = model_no_fuse(x)
    print(f"  无融合输出 / No-fusion output: {out2.shape}")

    print("\n[Test 2] Focal Loss")
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(out, torch.tensor([0, 3]))
    print(f"  Loss = {loss.item():.4f}")

    print("\n[Test 3] 手工特征提取 / Handcrafted feature extraction")
    fake_wm = np.zeros((30, 30), dtype=np.uint8)
    fake_wm[10:20, 10:20] = 1
    fake_wm[12:15, 12:15] = 2
    feat = extract_spatial_features(fake_wm)
    print(f"  特征维度 / Feature dim: {feat.shape}")
    print(f"  特征值 / Feature values: {feat}")

    print("\n" + "=" * 60)
    print("  自测完成 / Self-test complete")
    print("=" * 60)
