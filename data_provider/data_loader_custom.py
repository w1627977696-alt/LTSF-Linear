"""
自定义数据集加载器，用于加载分离的训练/测试CSV文件。
本模块支持使用分离的训练和测试文件加载您自己的时间序列数据。

数据格式要求:
1. CSV文件，第一列为'date'（可选，可以用索引代替）
2. 其余列为特征列
3. 应指定目标列（用于S或MS模式）

CSV结构示例:
    date,feature1,feature2,target
    2020-01-01,1.0,2.0,3.0
    2020-01-02,1.1,2.1,3.1
    ...

或不带日期列（将使用索引作为日期）:
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
    用于加载分离的训练/测试CSV文件的自定义数据集类。
    
    参数:
    -----------
    root_path : str
        包含数据文件的根目录路径
    flag : str
        'train', 'val', 或 'test'，指定要加载的数据集
    size : list
        [seq_len, label_len, pred_len] - 模型的序列长度
    features : str
        'S': 单变量到单变量
        'M': 多变量到多变量
        'MS': 多变量到单变量
    train_data_path : str
        训练数据CSV的文件名
    test_data_path : str
        测试数据CSV的文件名
    target : str
        目标列名（用于S或MS模式）
    scale : bool
        是否对数据进行标准化
    val_ratio : float
        用于验证的训练数据比例（默认: 0.2）
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
        
        # 读取训练数据
        train_df = pd.read_csv(os.path.join(self.root_path, self.train_data_path))
        # 读取测试数据
        test_df = pd.read_csv(os.path.join(self.root_path, self.test_data_path))
        
        # 检查是否存在'date'列
        has_date = 'date' in train_df.columns.str.lower()
        
        # 获取列名（如果存在则排除date）
        if has_date:
            date_col = [c for c in train_df.columns if c.lower() == 'date'][0]
            cols = [c for c in train_df.columns if c.lower() != 'date']
        else:
            date_col = None
            cols = list(train_df.columns)
        
        # 根据模式选择特征
        if self.features == 'M' or self.features == 'MS':
            # 使用所有列作为特征
            df_train = train_df[cols]
            df_test = test_df[cols]
        elif self.features == 'S':
            # 仅使用目标列
            if self.target in cols:
                df_train = train_df[[self.target]]
                df_test = test_df[[self.target]]
            else:
                raise ValueError(f"在数据中未找到目标列 '{self.target}'。可用的列: {cols}")
        
        # 使用训练数据统计量进行标准化
        if self.scale:
            self.scaler.fit(df_train.values)
            train_data = self.scaler.transform(df_train.values)
            test_data = self.scaler.transform(df_test.values)
        else:
            train_data = df_train.values
            test_data = df_test.values
        
        # 将训练数据分为训练集和验证集
        num_train = int(len(train_data) * (1 - self.val_ratio))
        
        if self.flag == 'train':
            self.data_x = train_data[:num_train]
            self.data_y = train_data[:num_train]
        elif self.flag == 'val':
            # 对于验证集，需要在验证起点之前有seq_len个数据点
            # 以确保可以为所有验证样本构建完整的输入序列
            val_start = max(0, num_train - self.seq_len)
            self.data_x = train_data[val_start:]
            self.data_y = train_data[val_start:]
        else:  # test
            self.data_x = test_data
            self.data_y = test_data
        
        # 创建虚拟时间戳（DLinear不使用，但为了兼容性需要）
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
        """将标准化数据转换回原始尺度"""
        return self.scaler.inverse_transform(data)
    
    def get_scaler(self):
        """返回已拟合的标准化器供外部使用"""
        return self.scaler
