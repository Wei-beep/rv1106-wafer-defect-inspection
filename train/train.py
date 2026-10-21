#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WM-811K 晶圆图缺陷模式识别 - 完整训练脚本
Full Training Script for WM-811K Wafer Defect Pattern Recognition
==================================================================

功能 / Features:
    ✅ 加载WM-811K数据集，分层划分训练/验证/测试集
    ✅ 仅在训练集统计手工特征均值/标准差
    ✅ 轻量级CNN + 手工特征双分支融合
    ✅ Focal Loss + 类别权重 处理类别不均衡
    ✅ 扩充约束数据增强（旋转+翻转+平移+缩放+随机die缺失）
    ✅ ReduceLROnPlateau 自动降学习率（监控val_macro_f1）
    ✅ 每个epoch验证集评估 macro-F1、各类Precision/Recall
    ✅ 混淆矩阵打印（验证集和测试集）
    ✅ Early Stopping（监控val macro-F1）
    ✅ 保存最佳模型
    ✅ 测试集最终评估
    ✅ 导出ONNX模型 + 推理对齐校验
    ✅ 训练曲线可视化保存

使用方法 / Usage:
    python train.py --data ./data/LSWMD.pkl --epochs 60 --batch_size 64 --gamma 1.5
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wm811k_dataset import (
    DEFECT_CLASSES, NUM_CLASSES, TARGET_SIZE, FEATURE_DIM,
    load_wm811k, stratified_split, compute_feature_stats,
    WM811KDataset, get_class_weights, get_weighted_sampler,
    FocalLoss, WaferMapCNN, export_to_onnx,
)

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────
# 1. 参数配置
# ──────────────────────────────────────

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='WM-811K Wafer Defect Training')
    parser.add_argument('--data', type=str, default='./data/LSWMD.pkl',
                         help='数据集路径 / Dataset path')
    parser.add_argument('--epochs', type=int, default=60,
                         help='最大训练轮数 / Max training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                         help='批次大小 / Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                         help='学习率 / Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                         help='权重衰减 / Weight decay')
    parser.add_argument('--gamma', type=float, default=1.5,
                         help='Focal Loss gamma')
    parser.add_argument('--patience', type=int, default=15,
                         help='Early stopping耐心轮数')
    parser.add_argument('--num_workers', type=int, default=4,
                         help='数据加载线程数')
    parser.add_argument('--sampler', action='store_true', default=False,
                         help='开启加权采样器（默认关闭）')
    parser.add_argument('--no_fusion', action='store_true', default=False,
                         help='关闭特征融合（消融实验）')
    parser.add_argument('--output_dir', type=str, default='./output',
                         help='输出目录 / Output directory')
    parser.add_argument('--seed', type=int, default=42,
                         help='随机种子 / Random seed')
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────
# 2. 评估函数
# ──────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, use_features=True, criterion=None):
    """在给定数据集上评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if use_features:
            images, features, labels = batch
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(images, features)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

        if criterion is not None:
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            n_batches += 1

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    per_class_f1 = f1_score(all_labels, all_preds, average=None,
                             labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None,
                                     labels=list(range(NUM_CLASSES)), zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_f1': per_class_f1,
        'per_class_recall': per_class_recall,
        'predictions': all_preds,
        'labels': all_labels,
    }


def print_metrics(result: dict, title: str = "Evaluation"):
    """打印评估指标"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Loss:           {result['loss']:.4f}")
    print(f"  Accuracy:       {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"  Macro-F1:       {result['macro_f1']:.4f}")
    print(f"  Macro-Precision:{result['macro_precision']:.4f}")
    print(f"  Macro-Recall:   {result['macro_recall']:.4f}")
    print(f"\n  每类F1 / Per-class F1:")
    for i, c in enumerate(DEFECT_CLASSES):
        print(f"    {c:<12} F1={result['per_class_f1'][i]:.4f}  "
               f"Recall={result['per_class_recall'][i]:.4f}")


def plot_confusion_matrix(labels, preds, title, save_path):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(NUM_CLASSES),
           yticks=np.arange(NUM_CLASSES),
           xticklabels=DEFECT_CLASSES,
           yticklabels=DEFECT_CLASSES,
           title=f'{title}\nConfusion Matrix (normalized)',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm_norm.max() / 2.
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                     ha='center', va='center',
                     color='white' if cm_norm[i, j] > thresh else 'black',
                     fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  混淆矩阵已保存 / Confusion matrix saved: {save_path}")


def plot_training_curve(train_losses, val_losses, train_f1s, val_f1s, save_path):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_f1s, 'b-', label='Train Macro-F1', linewidth=1.5)
    ax2.plot(epochs, val_f1s, 'r-', label='Val Macro-F1', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro-F1')
    ax2.set_title('Training & Validation Macro-F1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线已保存 / Training curve saved: {save_path}")

# ──────────────────────────────────────
# 3. 主训练流程
# ──────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] 使用设备 / Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Step 1: 加载数据
    print(f"\n[Step 1] 加载数据集 / Loading dataset: {args.data}")
    all_data = load_wm811k(args.data)
    train_samples, val_samples, test_samples = stratified_split(all_data, seed=args.seed)

    # Step 2: 统计手工特征标准化参数（仅训练集）
    print(f"\n[Step 2] 统计手工特征均值/标准差（仅训练集）")
    feat_norm_path = os.path.join(args.output_dir, 'feature_norm.npz')
    feat_mean, feat_std = compute_feature_stats(train_samples, save_path=feat_norm_path)

    # Step 3: 创建Dataset和DataLoader
    print(f"\n[Step 3] 创建DataLoader / Creating DataLoaders")

    use_fusion = not args.no_fusion
    use_sampler = args.sampler

    train_ds = WM811KDataset(train_samples, augment=True,
                              use_features=use_fusion,
                              feature_mean=feat_mean, feature_std=feat_std)
    val_ds = WM811KDataset(val_samples, augment=False,
                            use_features=use_fusion,
                            feature_mean=feat_mean, feature_std=feat_std)
    test_ds = WM811KDataset(test_samples, augment=False,
                             use_features=use_fusion,
                             feature_mean=feat_mean, feature_std=feat_std)

    sampler = get_weighted_sampler(train_samples) if use_sampler else None

    if use_sampler:
        print(f"  [策略] 加权采样器: 开启 / Weighted sampler: ON")
    else:
        print(f"  [策略] 加权采样器: 关闭（仅用Focal Loss）/ Weighted sampler: OFF")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print(f"  训练批次 / Train batches: {len(train_loader)}")
    print(f"  验证批次 / Val batches: {len(val_loader)}")
    print(f"  测试批次 / Test batches: {len(test_loader)}")

    # Step 4: 初始化模型、损失、优化器
    print(f"\n[Step 4] 初始化模型 / Initializing model")
    model = WaferMapCNN(num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM,
                         use_fusion=use_fusion).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量 / Model parameters: {n_params:,}")

    # 损失函数
    class_weights_np = get_class_weights(train_samples)
    class_weights = torch.from_numpy(class_weights_np).float().to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    print(f"  损失 / Loss: Focal Loss (gamma={args.gamma}, with class weights)")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

    # ReduceLROnPlateau 自动降学习率（监控val_macro_f1）
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5,
        min_lr=1e-6, verbose=True
    )
    print(f"  优化器 / Optimizer: AdamW (lr={args.lr})")
    print(f"  学习率调度 / LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")

    # Step 5: 训练循环
    print(f"\n[Step 5] 开始训练 / Starting training (max {args.epochs} epochs)")
    print(f"  [策略] 特征融合: {'开启/ON' if use_fusion else '关闭/OFF'}")
    print(f"  [策略] Focal Loss: 开启/ON (gamma={args.gamma})")
    print(f"  [策略] Early stopping patience: {args.patience} epochs")

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # 训练
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            if use_fusion:
                images, features, labels = batch
                images = images.to(device)
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(images, features)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} "
                       f"| Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        # 训练集指标
        avg_train_loss = running_loss / len(train_loader)
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)

        # 验证
        val_result = evaluate(model, val_loader, device,
                               use_features=use_fusion, criterion=criterion)

        # ReduceLROnPlateau 步进（监控val_macro_f1）
        scheduler.step(val_result['macro_f1'])

        # 记录
        train_losses.append(avg_train_loss)
        val_losses.append(val_result['loss'])
        train_f1s.append(train_f1)
        val_f1s.append(val_result['macro_f1'])

        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
               f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f} | "
               f"Val Loss: {val_result['loss']:.4f}, Val F1: {val_result['macro_f1']:.4f} | "
               f"LR: {current_lr:.2e}")

        # Early Stopping 检查
        if val_result['macro_f1'] > best_val_f1:
            best_val_f1 = val_result['macro_f1']
            best_epoch = epoch
            patience_counter = 0

            best_model_path = os.path.join(args.output_dir, 'best_wafer.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_loss': val_result['loss'],
                'use_fusion': use_fusion,
            }, best_model_path)
            print(f"  ✅ 新最佳模型 / New best model saved! Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ 未提升 / No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n  🛑 Early stopping triggered at epoch {epoch}! "
                       f"Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
                break

    # Step 6: 测试集最终评估
    print(f"\n[Step 6] 测试集最终评估 / Final test set evaluation")
    best_model_path = os.path.join(args.output_dir, 'best_wafer.pt')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  已加载最佳模型 / Loaded best model from epoch {checkpoint['epoch']} "
           f"(Val F1: {checkpoint['val_f1']:.4f})")

    val_result = evaluate(model, val_loader, device,
                           use_features=use_fusion, criterion=criterion)
    print_metrics(val_result, "验证集 / Validation Set")

    test_result = evaluate(model, test_loader, device,
                            use_features=use_fusion, criterion=criterion)
    print_metrics(test_result, "测试集 / Test Set")

    print(f"\n  测试集分类报告 / Test Classification Report:")
    print(classification_report(test_result['labels'], test_result['predictions'],
                                 target_names=DEFECT_CLASSES, digits=4, zero_division=0))

    # Step 7: 绘制混淆矩阵
    print(f"\n[Step 7] 绘制混淆矩阵 / Plotting confusion matrices")
    plot_confusion_matrix(val_result['labels'], val_result['predictions'],
                           'Validation Set',
                           os.path.join(args.output_dir, 'confusion_matrix_val.png'))
    plot_confusion_matrix(test_result['labels'], test_result['predictions'],
                           'Test Set',
                           os.path.join(args.output_dir, 'confusion_matrix_test.png'))

    # Step 8: 绘制训练曲线
    print(f"\n[Step 8] 绘制训练曲线 / Plotting training curves")
    plot_training_curve(train_losses, val_losses, train_f1s, val_f1s,
                        os.path.join(args.output_dir, 'training_curve.png'))

    # Step 9: 导出ONNX
    print(f"\n[Step 9] 导出ONNX模型 / Exporting ONNX model")
    onnx_path = os.path.join(args.output_dir, 'wafer_model.onnx')
    export_to_onnx(model, onnx_path, use_features=use_fusion)

    # 总结
    print(f"\n{'='*60}")
    print(f"  训练完成 / Training Complete!")
    print(f"{'='*60}")
    print(f"  最佳轮次 / Best epoch: {best_epoch}")
    print(f"  最佳Val Macro-F1: {best_val_f1:.4f}")
    print(f"  测试集Macro-F1 / Test Macro-F1: {test_result['macro_f1']:.4f}")
    print(f"  测试集Accuracy / Test Accuracy: {test_result['accuracy']:.4f}")
    print(f"\n  输出文件 / Output files:")
    print(f"    - {feat_norm_path}")
    print(f"    - {best_model_path}")
    print(f"    - {onnx_path}")
    print(f"    - {os.path.join(args.output_dir, 'confusion_matrix_test.png')}")
    print(f"    - {os.path.join(args.output_dir, 'training_curve.png')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
