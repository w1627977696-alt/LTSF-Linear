# 自定义数据集时间序列预测指南

本指南帮助您使用自己的CSV数据集进行DLinear时间序列预测和可视化。

## 目录

1. [项目结构解析](#项目结构解析)
2. [数据格式要求](#数据格式要求)
3. [模型参数说明](#模型参数说明)
4. [快速开始](#快速开始)
5. [详细使用说明](#详细使用说明)
6. [可视化功能](#可视化功能)
7. [常见问题](#常见问题)

---

## 项目结构解析

```
LTSF-Linear/
├── models/
│   ├── DLinear.py          # DLinear模型核心实现
│   ├── NLinear.py          # NLinear模型
│   └── Linear.py           # 基础Linear模型
├── data_provider/
│   ├── data_loader.py      # 原始数据加载器
│   ├── data_loader_custom.py  # 自定义数据加载器（支持分离的train/test文件）
│   └── data_factory.py     # 数据工厂
├── exp/
│   ├── exp_main.py         # 主实验文件（训练、测试、预测）
│   └── exp_basic.py        # 基础实验类
├── utils/
│   ├── tools.py            # 工具函数（可视化、早停等）
│   └── metrics.py          # 评估指标
├── run_custom_prediction.py    # 自定义预测主脚本
├── visualize_predictions.py    # 可视化模块
└── CUSTOM_DATASET_GUIDE.md     # 本指南
```

### DLinear模型架构

DLinear模型通过分解时间序列为趋势(Trend)和季节(Seasonal)两个分量，分别用线性层进行预测：

```
输入 x: [Batch, seq_len, channels]
      ↓
  时序分解 (Moving Average)
      ↓
趋势分量 + 季节分量
      ↓
Linear层预测
      ↓
输出 y: [Batch, pred_len, channels]
```

---

## 数据格式要求

### CSV文件格式

您的训练和测试CSV文件应具有以下格式：

**格式1：带日期列**
```csv
date,feature1,feature2,target
2020-01-01 00:00:00,1.0,2.0,3.0
2020-01-01 01:00:00,1.1,2.1,3.1
2020-01-01 02:00:00,1.2,2.2,3.2
...
```

**格式2：无日期列**
```csv
feature1,feature2,target
1.0,2.0,3.0
1.1,2.1,3.1
1.2,2.2,3.2
...
```

### 重要说明

1. **目标列**: 默认情况下，最后一列被视为目标列。可通过`--target`参数指定其他列。
2. **特征列**: 除日期和目标外的所有列都是特征列。
3. **数据顺序**: 数据应按时间顺序排列。
4. **缺失值**: 不支持缺失值，请提前处理。

---

## 模型参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--seq_len` | 24 | 输入序列长度（回看窗口） |
| `--pred_len` | 10 | 预测长度（1步或多步） |
| `--features` | 'S' | 预测任务类型 |
| `--target` | 'target' | 目标列名 |
| `--individual` | False | 是否为每个通道使用独立的线性层 |

### Features参数说明

- **'S'** (单变量): 单变量预测单变量 - 只使用目标列进行预测
- **'M'** (多变量): 多变量预测多变量 - 使用所有特征预测所有特征
- **'MS'** (多变量→单变量): 多变量预测单变量 - 使用所有特征预测目标列

### 训练参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--batch_size` | 32 | 批次大小 |
| `--learning_rate` | 0.001 | 学习率 |
| `--train_epochs` | 50 | 训练轮数 |
| `--patience` | 5 | 早停耐心值 |
| `--val_ratio` | 0.2 | 验证集比例 |

---

## 快速开始

### 1. 准备数据

创建训练和测试CSV文件，放在同一目录下：

```
my_data/
├── train.csv    # 训练数据
└── test.csv     # 测试数据
```

### 2. 运行预测

**基本用法:**
```bash
python run_custom_prediction.py \
    --root_path ./my_data \
    --train_path train.csv \
    --test_path test.csv \
    --target value \
    --pred_len 10 \
    --seq_len 24
```

**1步预测:**
```bash
python run_custom_prediction.py \
    --root_path ./my_data \
    --train_path train.csv \
    --test_path test.csv \
    --pred_len 1 \
    --seq_len 24
```

**多变量预测:**
```bash
python run_custom_prediction.py \
    --root_path ./my_data \
    --train_path train.csv \
    --test_path test.csv \
    --features M \
    --pred_len 10
```

### 3. 查看结果

运行后，输出目录结构如下：
```
output/
├── checkpoint.pth           # 保存的最佳模型
├── training_history.csv     # 训练历史
├── training_history.png     # 训练曲线图
├── test_metrics.csv         # 测试指标
├── predictions.npy          # 预测结果
├── ground_truth.npy         # 真实值
└── visualizations/          # 可视化图表目录
    ├── prediction_sample_0.png
    ├── prediction_sample_1.png
    └── prediction_summary.png
```

---

## 详细使用说明

### 完整参数列表

```bash
python run_custom_prediction.py --help
```

**输出:**
```
usage: run_custom_prediction.py [-h] --train_path TRAIN_PATH --test_path TEST_PATH
                                 [--root_path ROOT_PATH] [--seq_len SEQ_LEN]
                                 [--pred_len PRED_LEN] [--features {S,M,MS}]
                                 [--target TARGET] [--individual]
                                 [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                                 [--train_epochs TRAIN_EPOCHS] [--patience PATIENCE]
                                 [--val_ratio VAL_RATIO] [--output_dir OUTPUT_DIR]
                                 [--num_vis_samples NUM_VIS_SAMPLES] [--seed SEED]
                                 [--no_scale]
```

### 示例场景

**场景1: 电力负荷预测（单变量）**
```bash
python run_custom_prediction.py \
    --root_path ./data \
    --train_path electricity_train.csv \
    --test_path electricity_test.csv \
    --target load \
    --features S \
    --seq_len 48 \
    --pred_len 24 \
    --output_dir ./results/electricity
```

**场景2: 股票价格预测（使用多特征）**
```bash
python run_custom_prediction.py \
    --root_path ./data \
    --train_path stock_train.csv \
    --test_path stock_test.csv \
    --target close_price \
    --features MS \
    --seq_len 30 \
    --pred_len 5 \
    --batch_size 64 \
    --learning_rate 0.0005
```

**场景3: 温度预测（小数据集）**
```bash
python run_custom_prediction.py \
    --root_path ./data \
    --train_path temp_train.csv \
    --test_path temp_test.csv \
    --target temperature \
    --features S \
    --seq_len 12 \
    --pred_len 1 \
    --batch_size 16 \
    --val_ratio 0.3
```

---

## 可视化功能

### 使用可视化模块

**方法1: 运行预测时自动生成**

运行`run_custom_prediction.py`时会自动生成可视化结果。

**方法2: 单独运行可视化脚本**

```bash
python visualize_predictions.py \
    --pred_path ./output/predictions.npy \
    --true_path ./output/ground_truth.npy \
    --output_dir ./my_visualizations
```

### 可视化输出说明

1. **prediction_sample_X.png**: 单个样本的预测对比图
   - 左图：标准化数值
   - 右图：原始尺度数值

2. **prediction_summary.png**: 综合汇总图
   - 多个样本对比
   - 预测vs真实散点图
   - 误差分布直方图
   - 各预测步的MSE

### 在Python中使用可视化类

```python
from visualize_predictions import PredictionVisualizer
import numpy as np

# 加载数据
preds = np.load('predictions.npy')
trues = np.load('ground_truth.npy')

# 创建可视化器
visualizer = PredictionVisualizer(preds, trues)

# 生成所有图表
visualizer.plot_all(output_dir='./my_plots')

# 或者单独生成特定图表
visualizer.plot_scatter(output_path='scatter.png')
visualizer.plot_error_distribution(output_path='errors.png')
visualizer.plot_summary(output_path='summary.png')
```

---

## 常见问题

### Q1: 数据量太少怎么办？

A: 减小`seq_len`和`batch_size`，增加`val_ratio`：
```bash
python run_custom_prediction.py \
    --seq_len 12 \
    --batch_size 8 \
    --val_ratio 0.3 \
    ...
```

### Q2: 如何处理多个目标变量？

A: 使用`features='M'`进行多变量预测：
```bash
python run_custom_prediction.py \
    --features M \
    ...
```

### Q3: 模型不收敛怎么办？

A: 尝试以下调整：
1. 减小学习率: `--learning_rate 0.0001`
2. 增加训练轮数: `--train_epochs 100`
3. 增加耐心值: `--patience 10`
4. 检查数据是否有异常值

### Q4: 如何只做1步预测？

A: 设置`pred_len=1`：
```bash
python run_custom_prediction.py \
    --pred_len 1 \
    ...
```

### Q5: 训练时内存不足？

A: 减小批次大小和序列长度：
```bash
python run_custom_prediction.py \
    --batch_size 16 \
    --seq_len 24 \
    ...
```

### Q6: 如何使用已训练的模型进行预测？

A: 可以加载保存的checkpoint进行预测：
```python
import torch
from models.DLinear import Model as DLinear

# 定义与训练时相同的参数
class Args:
    seq_len = 24
    pred_len = 10
    enc_in = 1
    individual = False

args = Args()
model = DLinear(args)
model.load_state_dict(torch.load('output/checkpoint.pth'))
model.eval()

# 进行预测
with torch.no_grad():
    # input_data shape: [batch, seq_len, channels]
    prediction = model(input_data)
```

---

## 更多资源

- 原始论文: [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
- GitHub仓库: [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

如有问题，欢迎提Issue！
