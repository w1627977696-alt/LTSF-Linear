#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module for Time Series Prediction Results

This module provides comprehensive visualization tools for comparing
predictions with ground truth values in time series forecasting.

Usage:
    1. As a standalone script:
       python visualize_predictions.py --pred_path predictions.npy --true_path ground_truth.npy
    
    2. As a module:
       from visualize_predictions import PredictionVisualizer
       visualizer = PredictionVisualizer(preds, trues)
       visualizer.plot_all()
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')


class PredictionVisualizer:
    """
    A comprehensive visualization class for time series predictions.
    
    Attributes:
    -----------
    preds : np.array
        Predicted values, shape: (num_samples, pred_len, num_features)
    trues : np.array
        Ground truth values, shape: (num_samples, pred_len, num_features)
    inputs : np.array, optional
        Input sequences, shape: (num_samples, seq_len, num_features)
    """
    
    def __init__(self, preds, trues, inputs=None, scaler=None):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        preds : np.array
            Predicted values
        trues : np.array
            Ground truth values
        inputs : np.array, optional
            Input sequences (for context visualization)
        scaler : object, optional
            Scaler object with inverse_transform method
        """
        self.preds = preds
        self.trues = trues
        self.inputs = inputs
        self.scaler = scaler
        
        # Ensure 3D arrays
        if self.preds.ndim == 2:
            self.preds = self.preds.reshape(self.preds.shape[0], self.preds.shape[1], 1)
        if self.trues.ndim == 2:
            self.trues = self.trues.reshape(self.trues.shape[0], self.trues.shape[1], 1)
        if self.inputs is not None and self.inputs.ndim == 2:
            self.inputs = self.inputs.reshape(self.inputs.shape[0], self.inputs.shape[1], 1)
        
        self.num_samples = self.preds.shape[0]
        self.pred_len = self.preds.shape[1]
        self.num_features = self.preds.shape[2]
        
    def plot_single_sample(self, sample_idx=0, feature_idx=-1, ax=None, 
                          show_input=True, title=None):
        """
        Plot prediction vs ground truth for a single sample.
        
        Parameters:
        -----------
        sample_idx : int
            Index of the sample to visualize
        feature_idx : int
            Index of the feature to plot (-1 for last feature)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        show_input : bool
            Whether to show input sequence
        title : str, optional
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        pred = self.preds[sample_idx, :, feature_idx]
        true = self.trues[sample_idx, :, feature_idx]
        
        if show_input and self.inputs is not None:
            inp = self.inputs[sample_idx, :, feature_idx]
            seq_len = len(inp)
            
            # Time axis
            t_input = np.arange(seq_len)
            t_pred = np.arange(seq_len, seq_len + self.pred_len)
            
            ax.plot(t_input, inp, 'b-', linewidth=2, label='Input Sequence')
            ax.plot(t_pred, true, 'g-', linewidth=2, label='Ground Truth')
            ax.plot(t_pred, pred, 'r--', linewidth=2, label='Prediction')
            ax.axvline(x=seq_len - 0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
            
            # Add shaded region for prediction zone
            ax.axvspan(seq_len - 0.5, seq_len + self.pred_len - 0.5, 
                      alpha=0.1, color='yellow', label='Prediction Zone')
        else:
            t = np.arange(self.pred_len)
            ax.plot(t, true, 'g-', linewidth=2, label='Ground Truth')
            ax.plot(t, pred, 'r--', linewidth=2, label='Prediction')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title or f'Sample {sample_idx}: Prediction vs Ground Truth', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_multiple_samples(self, num_samples=6, feature_idx=-1, output_path=None):
        """
        Plot multiple samples in a grid layout.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to visualize
        feature_idx : int
            Index of the feature to plot
        output_path : str, optional
            Path to save the figure
        """
        num_samples = min(num_samples, self.num_samples)
        
        # Calculate grid dimensions
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Select evenly spaced samples
        sample_indices = np.linspace(0, self.num_samples - 1, num_samples, dtype=int)
        
        for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
            self.plot_single_sample(sample_idx, feature_idx, ax, 
                                   show_input=(self.inputs is not None))
        
        # Hide unused subplots
        for ax in axes[num_samples:]:
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_scatter(self, feature_idx=-1, output_path=None):
        """
        Create scatter plot of predictions vs ground truth.
        
        Parameters:
        -----------
        feature_idx : int
            Index of the feature to plot
        output_path : str, optional
            Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        pred_flat = self.preds[:, :, feature_idx].flatten()
        true_flat = self.trues[:, :, feature_idx].flatten()
        
        ax.scatter(true_flat, pred_flat, alpha=0.3, s=20, c='blue')
        
        # Add perfect prediction line
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        ax.set_xlabel('Ground Truth', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.set_title(f'Prediction vs Ground Truth\n$R^2$ = {r2:.4f}', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_error_distribution(self, feature_idx=-1, output_path=None):
        """
        Plot error distribution histogram.
        
        Parameters:
        -----------
        feature_idx : int
            Index of the feature to analyze
        output_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        pred_flat = self.preds[:, :, feature_idx].flatten()
        true_flat = self.trues[:, :, feature_idx].flatten()
        errors = pred_flat - true_flat
        
        # Absolute error distribution
        ax1 = axes[0]
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=errors.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean: {errors.mean():.4f}')
        ax1.set_xlabel('Prediction Error', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Error Distribution\nMean: {errors.mean():.4f}, Std: {errors.std():.4f}', 
                     fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Relative error distribution (percentage)
        ax2 = axes[1]
        # Avoid division by zero
        mask = np.abs(true_flat) > 1e-6
        rel_errors = np.abs(errors[mask] / true_flat[mask]) * 100
        
        ax2.hist(rel_errors[rel_errors < np.percentile(rel_errors, 99)], 
                bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax2.axvline(x=np.median(rel_errors), color='green', linestyle='-', linewidth=2,
                   label=f'Median: {np.median(rel_errors):.2f}%')
        ax2.set_xlabel('Relative Error (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Relative Error Distribution\nMedian: {np.median(rel_errors):.2f}%', 
                     fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_error_by_step(self, feature_idx=-1, output_path=None):
        """
        Plot error metrics for each prediction step.
        
        Parameters:
        -----------
        feature_idx : int
            Index of the feature to analyze
        output_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        mse_by_step = []
        mae_by_step = []
        
        for step in range(self.pred_len):
            step_pred = self.preds[:, step, feature_idx]
            step_true = self.trues[:, step, feature_idx]
            mse_by_step.append(np.mean((step_pred - step_true) ** 2))
            mae_by_step.append(np.mean(np.abs(step_pred - step_true)))
        
        steps = np.arange(1, self.pred_len + 1)
        
        # MSE by step
        ax1 = axes[0]
        ax1.bar(steps, mse_by_step, alpha=0.7, edgecolor='black', color='steelblue')
        ax1.plot(steps, mse_by_step, 'ro-', markersize=6)
        ax1.set_xlabel('Prediction Step', fontsize=12)
        ax1.set_ylabel('MSE', fontsize=12)
        ax1.set_title('Mean Squared Error by Prediction Step', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # MAE by step
        ax2 = axes[1]
        ax2.bar(steps, mae_by_step, alpha=0.7, edgecolor='black', color='coral')
        ax2.plot(steps, mae_by_step, 'bo-', markersize=6)
        ax2.set_xlabel('Prediction Step', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title('Mean Absolute Error by Prediction Step', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_summary(self, feature_idx=-1, output_path=None):
        """
        Create a comprehensive summary plot with multiple views.
        
        Parameters:
        -----------
        feature_idx : int
            Index of the feature to analyze
        output_path : str, optional
            Path to save the figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        pred_flat = self.preds[:, :, feature_idx].flatten()
        true_flat = self.trues[:, :, feature_idx].flatten()
        errors = pred_flat - true_flat
        
        # Top row: Sample predictions (3 samples)
        sample_indices = [0, self.num_samples // 2, self.num_samples - 1]
        for idx, sample_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[0, idx])
            if sample_idx < self.num_samples:
                self.plot_single_sample(sample_idx, feature_idx, ax, 
                                       show_input=(self.inputs is not None),
                                       title=f'Sample {sample_idx}')
        
        # Middle left: Scatter plot
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_scatter.scatter(true_flat, pred_flat, alpha=0.2, s=10, c='blue')
        min_val, max_val = min(true_flat.min(), pred_flat.min()), max(true_flat.max(), pred_flat.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax_scatter.set_xlabel('Ground Truth')
        ax_scatter.set_ylabel('Prediction')
        ax_scatter.set_title('Prediction vs Ground Truth')
        ax_scatter.grid(True, alpha=0.3)
        
        # Middle center: Error distribution
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_hist.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax_hist.axvline(x=errors.mean(), color='green', linestyle='-', linewidth=2)
        ax_hist.set_xlabel('Error')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'Error Distribution\nμ={errors.mean():.4f}, σ={errors.std():.4f}')
        ax_hist.grid(True, alpha=0.3)
        
        # Middle right: Error by step
        ax_step = fig.add_subplot(gs[1, 2])
        mse_by_step = [np.mean((self.preds[:, s, feature_idx] - self.trues[:, s, feature_idx]) ** 2) 
                      for s in range(self.pred_len)]
        ax_step.bar(range(1, self.pred_len + 1), mse_by_step, alpha=0.7, color='coral')
        ax_step.set_xlabel('Prediction Step')
        ax_step.set_ylabel('MSE')
        ax_step.set_title('MSE by Prediction Step')
        ax_step.grid(True, alpha=0.3, axis='y')
        
        # Bottom row: Time series of predictions (continuous)
        ax_ts = fig.add_subplot(gs[2, :])
        
        # Show first N samples as continuous time series
        num_show = min(10, self.num_samples)
        pred_concat = self.preds[:num_show, :, feature_idx].flatten()
        true_concat = self.trues[:num_show, :, feature_idx].flatten()
        
        t = np.arange(len(pred_concat))
        ax_ts.plot(t, true_concat, 'g-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        ax_ts.plot(t, pred_concat, 'r--', linewidth=1.5, label='Prediction', alpha=0.8)
        
        # Add vertical lines to separate samples
        for i in range(1, num_show):
            ax_ts.axvline(x=i * self.pred_len - 0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax_ts.set_xlabel('Time Step')
        ax_ts.set_ylabel('Value')
        ax_ts.set_title(f'Continuous Prediction View (First {num_show} Samples)')
        ax_ts.legend(loc='best')
        ax_ts.grid(True, alpha=0.3)
        
        # Add metrics text box
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)
        
        metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        fig.text(0.02, 0.98, metrics_text, fontsize=10, verticalalignment='top',
                bbox=props, family='monospace')
        
        plt.suptitle('Time Series Prediction Summary', fontsize=16, y=1.02)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_all(self, output_dir='./visualizations', feature_idx=-1):
        """
        Generate all visualization plots and save to directory.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all plots
        feature_idx : int
            Index of the feature to analyze
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations...")
        print(f"Output directory: {output_dir}")
        print(f"Number of samples: {self.num_samples}")
        print(f"Prediction length: {self.pred_len}")
        print(f"Number of features: {self.num_features}")
        print("-" * 50)
        
        # Generate all plots
        self.plot_multiple_samples(num_samples=6, feature_idx=feature_idx,
                                  output_path=os.path.join(output_dir, 'multiple_samples.png'))
        
        self.plot_scatter(feature_idx=feature_idx,
                         output_path=os.path.join(output_dir, 'scatter_plot.png'))
        
        self.plot_error_distribution(feature_idx=feature_idx,
                                    output_path=os.path.join(output_dir, 'error_distribution.png'))
        
        self.plot_error_by_step(feature_idx=feature_idx,
                               output_path=os.path.join(output_dir, 'error_by_step.png'))
        
        self.plot_summary(feature_idx=feature_idx,
                         output_path=os.path.join(output_dir, 'summary.png'))
        
        print("-" * 50)
        print(f"All visualizations saved to: {output_dir}")
        
        # Calculate and print metrics
        pred_flat = self.preds[:, :, feature_idx].flatten()
        true_flat = self.trues[:, :, feature_idx].flatten()
        
        mse = np.mean((pred_flat - true_flat) ** 2)
        mae = np.mean(np.abs(pred_flat - true_flat))
        rmse = np.sqrt(mse)
        
        print("\nMetrics Summary:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")


def main():
    """Command-line interface for visualization."""
    parser = argparse.ArgumentParser(description='Visualize Time Series Predictions')
    
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to predictions numpy file')
    parser.add_argument('--true_path', type=str, required=True,
                       help='Path to ground truth numpy file')
    parser.add_argument('--input_path', type=str, default=None,
                       help='Path to input sequences numpy file (optional)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for plots')
    parser.add_argument('--feature_idx', type=int, default=-1,
                       help='Feature index to visualize (-1 for last feature)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    preds = np.load(args.pred_path)
    trues = np.load(args.true_path)
    inputs = np.load(args.input_path) if args.input_path else None
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Ground truth shape: {trues.shape}")
    if inputs is not None:
        print(f"Inputs shape: {inputs.shape}")
    
    # Create visualizer and generate plots
    visualizer = PredictionVisualizer(preds, trues, inputs)
    visualizer.plot_all(args.output_dir, args.feature_idx)


if __name__ == '__main__':
    main()
