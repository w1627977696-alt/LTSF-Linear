#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate sample time series data for testing the custom prediction script.

This script generates synthetic time series data with trend, seasonality, and noise
components for demonstration purposes.

Usage:
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
    Generate synthetic time series data.
    
    Parameters:
    -----------
    n_samples : int
        Number of time steps to generate
    freq : str
        Pandas frequency string
    start_date : str
        Starting date
    trend_slope : float
        Slope of linear trend
    seasonality_period : int
        Period of seasonality (e.g., 24 for daily pattern with hourly data)
    noise_std : float
        Standard deviation of noise
    n_features : int
        Number of additional features to generate
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with date, features, and target columns
    """
    # Generate time index
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    # Generate base signal with trend and seasonality
    t = np.arange(n_samples)
    
    # Trend component
    trend = trend_slope * t
    
    # Seasonality component (multiple patterns)
    daily_seasonality = 2 * np.sin(2 * np.pi * t / seasonality_period)
    weekly_seasonality = 1 * np.sin(2 * np.pi * t / (7 * seasonality_period))
    
    # Noise
    noise = np.random.normal(0, noise_std, n_samples)
    
    # Target: combination of components
    target = 10 + trend + daily_seasonality + weekly_seasonality + noise
    
    # Generate correlated features
    features = {}
    for i in range(n_features - 1):
        # Each feature has its own pattern but correlated with target
        feat_seasonality = np.sin(2 * np.pi * t / seasonality_period + i * np.pi / 4)
        feat_noise = np.random.normal(0, noise_std * 2, n_samples)
        features[f'feature_{i+1}'] = (
            5 + 0.5 * target + feat_seasonality + feat_noise
        )
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        **features,
        'target': target
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate sample time series data')
    parser.add_argument('--output_dir', type=str, default='./sample_data',
                       help='Output directory for sample data')
    parser.add_argument('--train_size', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--test_size', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--n_features', type=int, default=3,
                       help='Number of features (including target)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training data
    print(f"Generating training data ({args.train_size} samples)...")
    train_df = generate_time_series(
        n_samples=args.train_size,
        start_date='2020-01-01',
        n_features=args.n_features
    )
    
    # Generate test data (continuation from training)
    print(f"Generating test data ({args.test_size} samples)...")
    train_end_date = train_df['date'].iloc[-1]
    test_start_date = train_end_date + pd.Timedelta(hours=1)
    
    test_df = generate_time_series(
        n_samples=args.test_size,
        start_date=str(test_start_date),
        n_features=args.n_features
    )
    # Adjust test data to continue from training (shift to match trend)
    trend_offset = train_df['target'].iloc[-1] - 10
    test_df['target'] = test_df['target'] + trend_offset * 0.5
    for col in test_df.columns:
        if col.startswith('feature_'):
            test_df[col] = test_df[col] + trend_offset * 0.25
    
    # Save to CSV
    train_path = os.path.join(args.output_dir, 'train.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSample data generated successfully!")
    print(f"Training data: {train_path}")
    print(f"  - Samples: {len(train_df)}")
    print(f"  - Columns: {list(train_df.columns)}")
    print(f"\nTest data: {test_path}")
    print(f"  - Samples: {len(test_df)}")
    print(f"  - Columns: {list(test_df.columns)}")
    
    print("\n" + "="*60)
    print("To run prediction with this sample data, use:")
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
