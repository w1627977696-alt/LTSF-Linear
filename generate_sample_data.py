#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成用于测试自定义预测脚本的样本时间序列数据。

该脚本生成包含趋势、季节性和噪声成分的合成时间序列数据，用于演示目的。

使用方法:
    python generate_sample_data.py [--output_dir ./sample_data] [--train_size 500] [--test_size 100]
"""

import argparse
import os
import numpy as np
import pandas as pd


def generate_time_series(n_samples, freq='h', start_date='2020-01-01', 
                        trend_slope=0.01, seasonality_period=24, noise_std=0.1,
                        n_features=3):
    """
    生成合成时间序列数据。
    
    参数:
    -----------
    n_samples : int
        要生成的时间步数
    freq : str
        Pandas频率字符串
    start_date : str
        起始日期
    trend_slope : float
        线性趋势的斜率
    seasonality_period : int
        季节性周期（例如，对于每小时数据，24表示日模式）
    noise_std : float
        噪声的标准差
    n_features : int
        要生成的额外特征数量
    
    返回:
    --------
    pd.DataFrame
        包含日期、特征和目标列的DataFrame
    """
    # 生成时间索引
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    # 生成带有趋势和季节性的基础信号
    t = np.arange(n_samples)
    
    # 趋势成分
    trend = trend_slope * t
    
    # 季节性成分（多种模式）
    daily_seasonality = 2 * np.sin(2 * np.pi * t / seasonality_period)
    weekly_seasonality = 1 * np.sin(2 * np.pi * t / (7 * seasonality_period))
    
    # 噪声
    noise = np.random.normal(0, noise_std, n_samples)
    
    # 目标：各成分的组合
    target = 10 + trend + daily_seasonality + weekly_seasonality + noise
    
    # 生成相关特征
    features = {}
    for i in range(n_features - 1):
        # 每个特征有自己的模式但与目标相关
        feat_seasonality = np.sin(2 * np.pi * t / seasonality_period + i * np.pi / 4)
        feat_noise = np.random.normal(0, noise_std * 2, n_samples)
        features[f'feature_{i+1}'] = (
            5 + 0.5 * target + feat_seasonality + feat_noise
        )
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        **features,
        'target': target
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description='生成样本时间序列数据')
    parser.add_argument('--output_dir', type=str, default='./sample_data',
                       help='样本数据的输出目录')
    parser.add_argument('--train_size', type=int, default=500,
                       help='训练样本数量')
    parser.add_argument('--test_size', type=int, default=100,
                       help='测试样本数量')
    parser.add_argument('--n_features', type=int, default=3,
                       help='特征数量（包括目标）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重复性
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成训练数据
    print(f"正在生成训练数据（{args.train_size} 个样本）...")
    train_df = generate_time_series(
        n_samples=args.train_size,
        start_date='2020-01-01',
        n_features=args.n_features
    )
    
    # 生成测试数据（从训练数据延续）
    print(f"正在生成测试数据（{args.test_size} 个样本）...")
    train_end_date = train_df['date'].iloc[-1]
    test_start_date = train_end_date + pd.Timedelta(hours=1)
    
    test_df = generate_time_series(
        n_samples=args.test_size,
        start_date=str(test_start_date),
        n_features=args.n_features
    )
    # 调整测试数据以从训练数据延续（调整以匹配趋势）
    trend_offset = train_df['target'].iloc[-1] - 10
    test_df['target'] = test_df['target'] + trend_offset * 0.5
    for col in test_df.columns:
        if col.startswith('feature_'):
            test_df[col] = test_df[col] + trend_offset * 0.25
    
    # 保存为CSV
    train_path = os.path.join(args.output_dir, 'train.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n样本数据生成成功！")
    print(f"训练数据: {train_path}")
    print(f"  - 样本数: {len(train_df)}")
    print(f"  - 列名: {list(train_df.columns)}")
    print(f"\n测试数据: {test_path}")
    print(f"  - 样本数: {len(test_df)}")
    print(f"  - 列名: {list(test_df.columns)}")
    
    print("\n" + "="*60)
    print("使用此样本数据运行预测，请执行:")
    print("="*60)
    print(f"""
python run_custom_prediction.py \\
    --root_path {args.output_dir} \\
    --train_path train.csv \\
    --test_path test.csv \\
    --target target \\
    --seq_len 24 \\
    --pred_len 10 \\
    --output_dir ./output
""")


if __name__ == '__main__':
    main()
