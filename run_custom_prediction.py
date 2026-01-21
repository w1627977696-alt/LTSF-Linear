#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLinear 时间序列预测与可视化

本脚本提供完整的解决方案：
1. 加载自定义的训练/测试CSV数据集
2. 训练DLinear模型进行时间序列预测
3. 评估预测结果并生成可视化图表

使用方法:
    python run_custom_prediction.py --train_path train.csv --test_path test.csv --pred_len 10

数据格式要求:
    CSV文件包含列: [date (可选)], feature1, feature2, ..., target
    - 'date' 列是可选的
    - 目标列应该是最后一列（或通过 --target 指定）
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.DLinear import Model as DLinear
from data_provider.data_loader_custom import Dataset_Custom_Separate
from utils.metrics import metric


def set_seed(seed=2021):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Args:
    """配置类，用于存储模型和训练参数"""
    def __init__(self):
        # 模型参数
        self.seq_len = 24           # 输入序列长度（回看窗口）
        self.label_len = 12         # 标签序列长度（用于解码器，DLinear不使用）
        self.pred_len = 10          # 预测长度
        self.enc_in = 1             # 输入通道/特征数量
        self.individual = False     # 是否为每个通道使用独立的线性层
        
        # 训练参数
        self.batch_size = 32
        self.learning_rate = 0.001
        self.train_epochs = 50
        self.patience = 5           # 早停耐心值
        
        # 数据参数
        self.features = 'S'         # 'S': 单变量, 'M': 多变量, 'MS': 多变量→单变量
        self.target = 'target'      # 目标列名
        self.scale = True           # 是否标准化数据
        self.val_ratio = 0.2        # 验证集比例
        
        # 设备
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')


def get_data_loader(root_path, train_path, test_path, flag, args):
    """为训练/验证/测试集创建数据加载器"""
    dataset = Dataset_Custom_Separate(
        root_path=root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        train_data_path=train_path,
        test_data_path=test_path,
        target=args.target,
        scale=args.scale,
        val_ratio=args.val_ratio
    )
    
    shuffle = (flag == 'train')
    drop_last = (flag == 'train')
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last
    )
    
    return dataset, data_loader


def train_model(model, train_loader, val_loader, args, save_path):
    """使用早停机制训练DLinear模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print("开始训练")
    print(f"{'='*60}")
    print(f"设备: {args.device}")
    print(f"训练轮数: {args.train_epochs}, 批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"序列长度: {args.seq_len}, 预测长度: {args.pred_len}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.train_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = []
        
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # 获取预测部分
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss.append(loss.item())
        
        train_loss = np.mean(epoch_train_loss)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = []
        
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.float().to(args.device)
                batch_y = batch_y.float().to(args.device)
                
                outputs = model(batch_x)
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                
                loss = criterion(outputs, batch_y)
                epoch_val_loss.append(loss.item())
        
        val_loss = np.mean(epoch_val_loss) if epoch_val_loss else train_loss
        val_losses.append(val_loss)
        
        print(f"轮次 {epoch+1:3d}/{args.train_epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), save_path)
            print(f"  -> 最佳模型已保存! (验证损失: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n在第{epoch+1}轮触发早停")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, test_dataset, args):
    """在测试集上评估模型，返回预测值和真实值"""
    model.eval()
    
    preds = []
    trues = []
    inputs = []
    
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            
            outputs = model(batch_x)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())
            inputs.append(batch_x.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputs = np.concatenate(inputs, axis=0)
    
    # 计算指标
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    
    print(f"\n{'='*60}")
    print("测试结果")
    print(f"{'='*60}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"RSE:  {rse:.6f}")
    print(f"{'='*60}\n")
    
    return preds, trues, inputs, {'mse': mse, 'mae': mae, 'rmse': rmse, 'rse': rse}


def visualize_predictions(preds, trues, inputs, scaler, args, output_dir, num_samples=5):
    """
    生成预测与真实值对比的可视化图表
    
    参数:
    -----------
    preds : np.array
        预测值, 形状: (样本数, 预测长度, 特征数)
    trues : np.array
        真实值, 形状: (样本数, 预测长度, 特征数)
    inputs : np.array
        输入序列, 形状: (样本数, 序列长度, 特征数)
    scaler : StandardScaler
        用于逆变换的标准化器
    args : Args
        配置对象
    output_dir : str
        保存图表的目录
    num_samples : int
        要生成的样本图表数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择要可视化的样本
    total_samples = len(preds)
    if total_samples < num_samples:
        sample_indices = list(range(total_samples))
    else:
        # 均匀分布的样本
        sample_indices = np.linspace(0, total_samples-1, num_samples, dtype=int)
    
    print(f"正在生成 {len(sample_indices)} 个可视化图表...")
    
    for idx, sample_idx in enumerate(sample_indices):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 获取该样本的数据
        input_seq = inputs[sample_idx, :, -1]  # 最后一个特征（通常是目标）
        pred_seq = preds[sample_idx, :, -1]
        true_seq = trues[sample_idx, :, -1]
        
        # 图1: 标准化值
        ax1 = axes[0]
        time_input = np.arange(len(input_seq))
        time_pred = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
        
        ax1.plot(time_input, input_seq, 'b-', label='输入序列', linewidth=2)
        ax1.plot(time_pred, true_seq, 'g-', label='真实值', linewidth=2)
        ax1.plot(time_pred, pred_seq, 'r--', label='预测值', linewidth=2)
        ax1.axvline(x=len(input_seq)-0.5, color='gray', linestyle=':', alpha=0.7)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('标准化值')
        ax1.set_title(f'样本 {sample_idx}: 标准化预测 vs 真实值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 原始尺度值（如果提供了scaler）
        ax2 = axes[1]
        if scaler is not None:
            # 为逆变换创建数组
            n_features = preds.shape[-1]
            
            # 逆变换输入
            input_orig = scaler.inverse_transform(
                inputs[sample_idx].reshape(-1, n_features)
            )[:, -1]
            
            # 逆变换预测值和真实值
            pred_orig = scaler.inverse_transform(
                preds[sample_idx].reshape(-1, n_features)
            )[:, -1]
            true_orig = scaler.inverse_transform(
                trues[sample_idx].reshape(-1, n_features)
            )[:, -1]
            
            ax2.plot(time_input, input_orig, 'b-', label='输入序列', linewidth=2)
            ax2.plot(time_pred, true_orig, 'g-', label='真实值', linewidth=2)
            ax2.plot(time_pred, pred_orig, 'r--', label='预测值', linewidth=2)
            ax2.axvline(x=len(input_orig)-0.5, color='gray', linestyle=':', alpha=0.7)
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('原始值')
            ax2.set_title(f'样本 {sample_idx}: 原始尺度预测 vs 真实值')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '标准化器不可用\n无法进行逆变换',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('原始尺度（不可用）')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_sample_{sample_idx}.png'), dpi=150)
        plt.close()
    
    # 生成汇总图
    generate_summary_plot(preds, trues, inputs, scaler, args, output_dir)
    
    print(f"可视化结果已保存至: {output_dir}")


def generate_summary_plot(preds, trues, inputs, scaler, args, output_dir):
    """生成展示整体预测性能的汇总图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 第一个样本 - 详细视图
    ax1 = axes[0, 0]
    sample_idx = 0
    input_seq = inputs[sample_idx, :, -1]
    pred_seq = preds[sample_idx, :, -1]
    true_seq = trues[sample_idx, :, -1]
    
    time_input = np.arange(len(input_seq))
    time_pred = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
    
    ax1.plot(time_input, input_seq, 'b-', label='输入', linewidth=2)
    ax1.plot(time_pred, true_seq, 'g-', label='真实值', linewidth=2)
    ax1.plot(time_pred, pred_seq, 'r--', label='预测值', linewidth=2)
    ax1.axvline(x=len(input_seq)-0.5, color='gray', linestyle=':', alpha=0.7)
    ax1.set_title('样本预测（第一个测试样本）')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('标准化值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 预测 vs 真实值散点图
    ax2 = axes[0, 1]
    all_preds = preds[:, :, -1].flatten()
    all_trues = trues[:, :, -1].flatten()
    
    ax2.scatter(all_trues, all_preds, alpha=0.3, s=10)
    min_val = min(all_trues.min(), all_preds.min())
    max_val = max(all_trues.max(), all_preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('预测值')
    ax2.set_title('预测 vs 真实值（所有样本）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 预测误差分布
    ax3 = axes[1, 0]
    errors = all_preds - all_trues
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('预测误差')
    ax3.set_ylabel('频率')
    ax3.set_title(f'预测误差分布\n均值: {errors.mean():.4f}, 标准差: {errors.std():.4f}')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 各预测步的误差
    ax4 = axes[1, 1]
    step_mse = []
    for step in range(args.pred_len):
        step_pred = preds[:, step, -1]
        step_true = trues[:, step, -1]
        step_mse.append(np.mean((step_pred - step_true) ** 2))
    
    ax4.bar(range(1, args.pred_len + 1), step_mse, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('预测步')
    ax4.set_ylabel('MSE')
    ax4.set_title('各预测步的MSE')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_summary.png'), dpi=150)
    plt.close()
    
    print(f"汇总图已保存至: {os.path.join(output_dir, 'prediction_summary.png')}")


def main():
    parser = argparse.ArgumentParser(description='DLinear 时间序列预测与可视化')
    
    # 数据路径
    parser.add_argument('--root_path', type=str, default='./', 
                        help='包含数据文件的根目录')
    parser.add_argument('--train_path', type=str, required=True,
                        help='训练数据CSV文件路径（相对于root_path）')
    parser.add_argument('--test_path', type=str, required=True,
                        help='测试数据CSV文件路径（相对于root_path）')
    
    # 模型参数
    parser.add_argument('--seq_len', type=int, default=24,
                        help='输入序列长度（回看窗口）')
    parser.add_argument('--pred_len', type=int, default=10,
                        help='预测长度（1步或多步）')
    parser.add_argument('--features', type=str, default='S',
                        choices=['S', 'M', 'MS'],
                        help='S: 单变量, M: 多变量, MS: 多变量→单变量')
    parser.add_argument('--target', type=str, default='target',
                        help='目标列名')
    parser.add_argument('--individual', action='store_true',
                        help='为每个通道使用独立的线性层')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='从训练数据中划分的验证集比例')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='模型和可视化结果的输出目录')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='要可视化的样本数量')
    
    # 其他
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--no_scale', action='store_true',
                        help='禁用数据标准化')
    
    cli_args = parser.parse_args()
    
    # 设置随机种子
    set_seed(cli_args.seed)
    
    # 创建Args对象
    args = Args()
    args.seq_len = cli_args.seq_len
    # label_len不被DLinear使用，但数据加载器需要
    # 设置为seq_len的一半是LTSF模型的常见惯例
    args.label_len = cli_args.seq_len // 2
    args.pred_len = cli_args.pred_len
    args.features = cli_args.features
    args.target = cli_args.target
    args.individual = cli_args.individual
    args.batch_size = cli_args.batch_size
    args.learning_rate = cli_args.learning_rate
    args.train_epochs = cli_args.train_epochs
    args.patience = cli_args.patience
    args.val_ratio = cli_args.val_ratio
    args.scale = not cli_args.no_scale
    
    # 创建输出目录
    output_dir = cli_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    print(f"\n{'='*60}")
    print("DLinear 时间序列预测")
    print(f"{'='*60}")
    print(f"训练数据: {cli_args.train_path}")
    print(f"测试数据: {cli_args.test_path}")
    print(f"序列长度: {args.seq_len}")
    print(f"预测长度: {args.pred_len}")
    print(f"特征模式: {args.features}")
    print(f"目标列: {args.target}")
    print(f"{'='*60}\n")
    
    # 加载数据并确定特征数量
    train_dataset, train_loader = get_data_loader(
        cli_args.root_path, cli_args.train_path, cli_args.test_path, 'train', args
    )
    val_dataset, val_loader = get_data_loader(
        cli_args.root_path, cli_args.train_path, cli_args.test_path, 'val', args
    )
    test_dataset, test_loader = get_data_loader(
        cli_args.root_path, cli_args.train_path, cli_args.test_path, 'test', args
    )
    
    # 从数据中获取输入通道数
    sample_x, _, _, _ = train_dataset[0]
    args.enc_in = sample_x.shape[-1]
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"输入通道数: {args.enc_in}")
    
    # 创建模型
    model = DLinear(args).float().to(args.device)
    print(f"\n模型已创建: DLinear")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, args, checkpoint_path
    )
    
    # 评估模型
    preds, trues, inputs, metrics = evaluate_model(
        model, test_loader, test_dataset, args
    )
    
    # 获取scaler用于逆变换
    scaler = test_dataset.get_scaler() if args.scale else None
    
    # 生成可视化
    visualize_predictions(
        preds, trues, inputs, scaler, args, vis_dir, cli_args.num_vis_samples
    )
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)
    
    # 保存预测结果
    np.save(os.path.join(output_dir, 'predictions.npy'), preds)
    np.save(os.path.join(output_dir, 'ground_truth.npy'), trues)
    
    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练历史')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"模型已保存至: {checkpoint_path}")
    print(f"可视化结果已保存至: {vis_dir}")
    print(f"指标已保存至: {os.path.join(output_dir, 'test_metrics.csv')}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
