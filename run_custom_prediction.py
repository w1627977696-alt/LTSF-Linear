#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLinear Time Series Prediction with Visualization

This script provides a complete solution for:
1. Loading custom train/test CSV datasets
2. Training DLinear model for time series prediction
3. Evaluating predictions and generating visualizations

Author: LTSF-Linear Custom Script
Usage:
    python run_custom_prediction.py --train_path train.csv --test_path test.csv --pred_len 10

Data Format Requirements:
    CSV files with columns: [date (optional)], feature1, feature2, ..., target
    - The 'date' column is optional
    - Target column should be the last column (or specified with --target)
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.DLinear import Model as DLinear
from data_provider.data_loader_custom import Dataset_Custom_Separate
from utils.metrics import metric


def set_seed(seed=2021):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Args:
    """Configuration class to hold model and training parameters."""
    def __init__(self):
        # Model parameters
        self.seq_len = 24           # Input sequence length (lookback window)
        self.label_len = 12         # Label sequence length (for decoder, not used in DLinear)
        self.pred_len = 10          # Prediction length
        self.enc_in = 1             # Number of input channels/features
        self.individual = False     # Whether to use individual linear layers for each channel
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.train_epochs = 50
        self.patience = 5           # Early stopping patience
        
        # Data parameters
        self.features = 'S'         # 'S': univariate, 'M': multivariate, 'MS': multivariate->univariate
        self.target = 'target'      # Target column name
        self.scale = True           # Whether to standardize data
        self.val_ratio = 0.2        # Validation split ratio
        
        # Device
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')


def get_data_loader(root_path, train_path, test_path, flag, args):
    """Create data loader for train/val/test sets."""
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
    """Train the DLinear model with early stopping."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.train_epochs}, Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Sequence Length: {args.seq_len}, Prediction Length: {args.pred_len}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.train_epochs):
        # Training phase
        model.train()
        epoch_train_loss = []
        
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Get prediction part
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss.append(loss.item())
        
        train_loss = np.mean(epoch_train_loss)
        train_losses.append(train_loss)
        
        # Validation phase
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
        
        print(f"Epoch {epoch+1:3d}/{args.train_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, test_dataset, args):
    """Evaluate model on test set and return predictions and ground truth."""
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
    
    # Calculate metrics
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"RSE:  {rse:.6f}")
    print(f"{'='*60}\n")
    
    return preds, trues, inputs, {'mse': mse, 'mae': mae, 'rmse': rmse, 'rse': rse}


def visualize_predictions(preds, trues, inputs, scaler, args, output_dir, num_samples=5):
    """
    Generate visualization plots comparing predictions with ground truth.
    
    Parameters:
    -----------
    preds : np.array
        Predicted values, shape: (num_samples, pred_len, num_features)
    trues : np.array
        Ground truth values, shape: (num_samples, pred_len, num_features)
    inputs : np.array
        Input sequences, shape: (num_samples, seq_len, num_features)
    scaler : StandardScaler
        Fitted scaler for inverse transformation
    args : Args
        Configuration object
    output_dir : str
        Directory to save plots
    num_samples : int
        Number of sample plots to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples to visualize
    total_samples = len(preds)
    if total_samples < num_samples:
        sample_indices = list(range(total_samples))
    else:
        # Evenly spaced samples
        sample_indices = np.linspace(0, total_samples-1, num_samples, dtype=int)
    
    print(f"Generating {len(sample_indices)} visualization plots...")
    
    for idx, sample_idx in enumerate(sample_indices):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get data for this sample
        input_seq = inputs[sample_idx, :, -1]  # Last feature (usually target)
        pred_seq = preds[sample_idx, :, -1]
        true_seq = trues[sample_idx, :, -1]
        
        # Plot 1: Normalized values
        ax1 = axes[0]
        time_input = np.arange(len(input_seq))
        time_pred = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
        
        ax1.plot(time_input, input_seq, 'b-', label='Input Sequence', linewidth=2)
        ax1.plot(time_pred, true_seq, 'g-', label='Ground Truth', linewidth=2)
        ax1.plot(time_pred, pred_seq, 'r--', label='Prediction', linewidth=2)
        ax1.axvline(x=len(input_seq)-0.5, color='gray', linestyle=':', alpha=0.7)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Normalized Value')
        ax1.set_title(f'Sample {sample_idx}: Normalized Prediction vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Original scale values (if scaler is provided)
        ax2 = axes[1]
        if scaler is not None:
            # Create dummy arrays for inverse transform
            n_features = preds.shape[-1]
            
            # Inverse transform input
            input_orig = scaler.inverse_transform(
                inputs[sample_idx].reshape(-1, n_features)
            )[:, -1]
            
            # Inverse transform predictions and true values
            pred_orig = scaler.inverse_transform(
                preds[sample_idx].reshape(-1, n_features)
            )[:, -1]
            true_orig = scaler.inverse_transform(
                trues[sample_idx].reshape(-1, n_features)
            )[:, -1]
            
            ax2.plot(time_input, input_orig, 'b-', label='Input Sequence', linewidth=2)
            ax2.plot(time_pred, true_orig, 'g-', label='Ground Truth', linewidth=2)
            ax2.plot(time_pred, pred_orig, 'r--', label='Prediction', linewidth=2)
            ax2.axvline(x=len(input_orig)-0.5, color='gray', linestyle=':', alpha=0.7)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Original Value')
            ax2.set_title(f'Sample {sample_idx}: Original Scale Prediction vs Ground Truth')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Scaler not available\nfor inverse transform',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Original Scale (Not Available)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_sample_{sample_idx}.png'), dpi=150)
        plt.close()
    
    # Generate summary plot
    generate_summary_plot(preds, trues, inputs, scaler, args, output_dir)
    
    print(f"Visualizations saved to: {output_dir}")


def generate_summary_plot(preds, trues, inputs, scaler, args, output_dir):
    """Generate a summary plot showing overall prediction performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: First sample - detailed view
    ax1 = axes[0, 0]
    sample_idx = 0
    input_seq = inputs[sample_idx, :, -1]
    pred_seq = preds[sample_idx, :, -1]
    true_seq = trues[sample_idx, :, -1]
    
    time_input = np.arange(len(input_seq))
    time_pred = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
    
    ax1.plot(time_input, input_seq, 'b-', label='Input', linewidth=2)
    ax1.plot(time_pred, true_seq, 'g-', label='Ground Truth', linewidth=2)
    ax1.plot(time_pred, pred_seq, 'r--', label='Prediction', linewidth=2)
    ax1.axvline(x=len(input_seq)-0.5, color='gray', linestyle=':', alpha=0.7)
    ax1.set_title('Sample Prediction (First Test Sample)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Normalized Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction vs Ground Truth scatter plot
    ax2 = axes[0, 1]
    all_preds = preds[:, :, -1].flatten()
    all_trues = trues[:, :, -1].flatten()
    
    ax2.scatter(all_trues, all_preds, alpha=0.3, s=10)
    min_val = min(all_trues.min(), all_preds.min())
    max_val = max(all_trues.max(), all_preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Prediction')
    ax2.set_title('Prediction vs Ground Truth (All Samples)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction error distribution
    ax3 = axes[1, 0]
    errors = all_preds - all_trues
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Prediction Error Distribution\nMean: {errors.mean():.4f}, Std: {errors.std():.4f}')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error by prediction step
    ax4 = axes[1, 1]
    step_mse = []
    for step in range(args.pred_len):
        step_pred = preds[:, step, -1]
        step_true = trues[:, step, -1]
        step_mse.append(np.mean((step_pred - step_true) ** 2))
    
    ax4.bar(range(1, args.pred_len + 1), step_mse, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Prediction Step')
    ax4.set_ylabel('MSE')
    ax4.set_title('MSE by Prediction Step')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_summary.png'), dpi=150)
    plt.close()
    
    print(f"Summary plot saved to: {os.path.join(output_dir, 'prediction_summary.png')}")


def main():
    parser = argparse.ArgumentParser(description='DLinear Time Series Prediction with Visualization')
    
    # Data paths
    parser.add_argument('--root_path', type=str, default='./', 
                        help='Root directory containing data files')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Training data CSV file path (relative to root_path)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Test data CSV file path (relative to root_path)')
    
    # Model parameters
    parser.add_argument('--seq_len', type=int, default=24,
                        help='Input sequence length (lookback window)')
    parser.add_argument('--pred_len', type=int, default=10,
                        help='Prediction length (1-step or multi-step)')
    parser.add_argument('--features', type=str, default='S',
                        choices=['S', 'M', 'MS'],
                        help='S: univariate, M: multivariate, MS: multivariate->univariate')
    parser.add_argument('--target', type=str, default='target',
                        help='Target column name')
    parser.add_argument('--individual', action='store_true',
                        help='Use individual linear layers for each channel')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio from training data')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for model and visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    # Other
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--no_scale', action='store_true',
                        help='Disable data standardization')
    
    cli_args = parser.parse_args()
    
    # Set seed
    set_seed(cli_args.seed)
    
    # Create Args object
    args = Args()
    args.seq_len = cli_args.seq_len
    # label_len is not used by DLinear but required for data loader compatibility
    # Setting to half of seq_len is a common convention in LTSF models
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
    
    # Create output directory
    output_dir = cli_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    print(f"\n{'='*60}")
    print("DLinear Time Series Prediction")
    print(f"{'='*60}")
    print(f"Train Data: {cli_args.train_path}")
    print(f"Test Data:  {cli_args.test_path}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Features Mode: {args.features}")
    print(f"Target Column: {args.target}")
    print(f"{'='*60}\n")
    
    # Load data and determine number of features
    train_dataset, train_loader = get_data_loader(
        cli_args.root_path, cli_args.train_path, cli_args.test_path, 'train', args
    )
    val_dataset, val_loader = get_data_loader(
        cli_args.root_path, cli_args.train_path, cli_args.test_path, 'val', args
    )
    test_dataset, test_loader = get_data_loader(
        cli_args.root_path, cli_args.train_path, cli_args.test_path, 'test', args
    )
    
    # Get number of input channels from data
    sample_x, _, _, _ = train_dataset[0]
    args.enc_in = sample_x.shape[-1]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Input channels: {args.enc_in}")
    
    # Create model
    model = DLinear(args).float().to(args.device)
    print(f"\nModel created: DLinear")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, args, checkpoint_path
    )
    
    # Evaluate model
    preds, trues, inputs, metrics = evaluate_model(
        model, test_loader, test_dataset, args
    )
    
    # Get scaler for inverse transform
    scaler = test_dataset.get_scaler() if args.scale else None
    
    # Generate visualizations
    visualize_predictions(
        preds, trues, inputs, scaler, args, vis_dir, cli_args.num_vis_samples
    )
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)
    
    # Save predictions
    np.save(os.path.join(output_dir, 'predictions.npy'), preds)
    np.save(os.path.join(output_dir, 'ground_truth.npy'), trues)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Visualizations saved to: {vis_dir}")
    print(f"Metrics saved to: {os.path.join(output_dir, 'test_metrics.csv')}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
