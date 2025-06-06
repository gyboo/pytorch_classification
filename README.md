# 深度学习图像分类训练框架

这是一个基于 PyTorch 的图像分类训练框架，支持多种预训练模型，具有自动日志记录和可视化功能。框架设计支持多次训练对比，并在网络不稳定时提供本地日志替代方案。

## 主要功能

- 支持多种模型架构（如 ViT、ResNet 等）
- 自动数据划分与加载
- 灵活的配置文件管理
- 训练过程可视化与记录
- 支持多次训练并比较结果
- 自动处理 Wandb 连接问题，提供本地日志替代方案

## 安装依赖

首先克隆仓库并安装必要的依赖：


```bash
git clone https://github.com/yourusername/pytorch_classification.git
cd pytorch_classification
```

## 使用方法

### 配置训练参数

修改配置文件 `docs/config.yaml` 来设置训练参数，例如：

  

```yaml
# 数据配置
connected_csv_path: "path/to/your/data.csv"
target_label: "class"
index_name: "id"
train_radio: 0.8

# 模型配置
model:
  name: "vit_base_patch16_224"
  pretrain: true
  pretrain_weight_path: "path/to/pretrained_weights.pth"

# 训练配置
epochs: 50
batch_size: 32

# 优化器配置
optimizer:
  name: "Adam"
  learning_rate: 0.001
  step_size: 10
  gamma: 0.1
```

### 运行训练

执行以下命令开始训练：


```bash
python train.py --train_time 5 --output_path results --config_file docs/config.yaml
```

  

参数说明：

  

- `--train_time`: 训练次数（默认为 5）
- `--output_path`: 输出结果保存路径（默认为 "output_path"）
- `--config_file`: 配置文件路径（默认为 "docs/config.yaml"）

### 结果查看

训练结果将保存在指定的输出路径中，包括：

  

- 每次训练的最佳模型权重
- 训练和验证数据的 CSV 文件
- 训练历史记录（损失和准确率）
- 可视化图表（损失曲线和准确率曲线）

  

如果 Wandb 可用，训练过程也会同步到 Wandb 平台。

## 代码结构


```plaintext
pytorch_classification/
├── dataset/                # 数据集处理模块
├── model/                  # 模型定义模块
├── docs/                   # 配置文件
├── output_path/            # 训练输出结果
│   ├── logs/               # 日志和图表
│   └── time0/              # 第一次训练结果
├── train.py                # 主训练脚本
└── requirements.txt        # 依赖文件
```

## 日志记录

框架支持两种日志记录方式：

  

1. **Wandb**：自动记录训练过程并上传到 Wandb 平台
2. **本地日志**：当 Wandb 连接失败时，自动切换到本地日志记录，生成 CSV 文件和可视化图表

  

本地日志包含：

  

- 每次训练的详细历史记录
- 所有训练的比较图表
- 损失曲线（训练和验证）
- 准确率曲线


## 许可证

本项目采用 MIT 许可证 - 详情见 LICENSE 文件

## 联系信息

如果你有任何问题或建议，请联系：your.email@example.com

## 致谢

感谢所有贡献者和开源社区的支持！
