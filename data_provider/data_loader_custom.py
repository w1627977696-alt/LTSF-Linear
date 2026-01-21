"""
Custom Dataset Loader for separate train/test CSV files.
This module supports loading your own time series data with separate train and test files.

Data Format Requirements:
1. CSV files with the first column as 'date' (optional, can be index)
2. Remaining columns are feature columns
3. The target column should be specified (for S or MS mode)

Example CSV structure:
    date,feature1,feature2,target
    2020-01-01,1.0,2.0,3.0
    2020-01-02,1.1,2.1,3.1
    ...

Or without date column (will use index as date):
    feature1,feature2,target
    1.0,2.0,3.0
    1.1,2.1,3.1
    ...
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom_Separate(Dataset):
    """
    Custom Dataset class for loading separate train/test CSV files.
    
    Parameters:
    -----------
    root_path : str
        Root directory path containing the data files
    flag : str
        'train', 'val', or 'test' to specify which dataset to load
    size : list
        [seq_len, label_len, pred_len] - sequence lengths for the model
    features : str
        'S': univariate-to-univariate
        'M': multivariate-to-multivariate
        'MS': multivariate-to-univariate
    train_data_path : str
        Filename for training data CSV
    test_data_path : str
        Filename for test data CSV  
    target : str
        Target column name (for S or MS mode)
    scale : bool
        Whether to standardize the data
    val_ratio : float
        Ratio of training data to use for validation (default: 0.2)
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='S', train_data_path='train.csv', test_data_path='test.csv',
                 target='target', scale=True, val_ratio=0.2, **kwargs):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 10
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        
        self.features = features
        self.target = target
        self.scale = scale
        self.val_ratio = val_ratio
        
        self.root_path = root_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # Read training data
        train_df = pd.read_csv(os.path.join(self.root_path, self.train_data_path))
        # Read test data
        test_df = pd.read_csv(os.path.join(self.root_path, self.test_data_path))
        
        # Check if 'date' column exists
        has_date = 'date' in train_df.columns.str.lower()
        
        # Get column names (excluding date if present)
        if has_date:
            date_col = [c for c in train_df.columns if c.lower() == 'date'][0]
            cols = [c for c in train_df.columns if c.lower() != 'date']
        else:
            date_col = None
            cols = list(train_df.columns)
        
        # Select features based on mode
        if self.features == 'M' or self.features == 'MS':
            # Use all columns as features
            df_train = train_df[cols]
            df_test = test_df[cols]
        elif self.features == 'S':
            # Use only target column
            if self.target in cols:
                df_train = train_df[[self.target]]
                df_test = test_df[[self.target]]
            else:
                raise ValueError(f"Target column '{self.target}' not found in data. Available columns: {cols}")
        
        # Standardize using training data statistics
        if self.scale:
            self.scaler.fit(df_train.values)
            train_data = self.scaler.transform(df_train.values)
            test_data = self.scaler.transform(df_test.values)
        else:
            train_data = df_train.values
            test_data = df_test.values
        
        # Split training data into train and validation
        num_train = int(len(train_data) * (1 - self.val_ratio))
        
        if self.flag == 'train':
            self.data_x = train_data[:num_train]
            self.data_y = train_data[:num_train]
        elif self.flag == 'val':
            # For validation, we need seq_len data points before the validation start
            # to ensure we can construct complete input sequences for all validation samples
            val_start = max(0, num_train - self.seq_len)
            self.data_x = train_data[val_start:]
            self.data_y = train_data[val_start:]
        else:  # test
            self.data_x = test_data
            self.data_y = test_data
        
        # Create dummy time stamps (not used by DLinear but needed for compatibility)
        self.data_stamp = np.zeros((len(self.data_x), 4))
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end] if r_end <= len(self.data_stamp) else np.zeros((self.label_len + self.pred_len, 4))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """Convert standardized data back to original scale."""
        return self.scaler.inverse_transform(data)
    
    def get_scaler(self):
        """Return the fitted scaler for external use."""
        return self.scaler
