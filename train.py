# 公共依赖
import os
import yaml
import torch
import random
import argparse
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time
from socket import timeout

warnings.filterwarnings("ignore")

# 渐度依赖
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# 模型依赖
from model.vit import create_vit
from model.Resnet18 import resnet18

# 数据依赖
from dataset.generate_dataset import BaseDataset

# 优化依赖
from torch import optim

# Wandb依赖
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleLogger:
    """简单的训练记录器，替代Wandb，支持记录多次训练"""

    def __init__(self, log_dir, config=None):
        self.log_dir = log_dir
        self.config = config
        self.all_histories = {}  # 存储所有训练的历史数据

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 保存配置
        if config:
            config_path = os.path.join(log_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

    def add_run(self, run_name):
        """添加一个新的训练运行"""
        self.all_histories[run_name] = []
        return self

    def log(self, run_name, metrics):
        """记录特定训练运行的指标"""
        if run_name not in self.all_histories:
            self.add_run(run_name)
        self.all_histories[run_name].append(metrics)

    def save_history(self, run_name):
        """保存特定训练运行的历史记录到CSV"""
        if run_name not in self.all_histories:
            print(f"运行 {run_name} 不存在")
            return

        history_df = pd.DataFrame(self.all_histories[run_name])
        history_path = os.path.join(self.log_dir, f"{run_name}_history.csv")
        history_df.to_csv(history_path, index=False)

    def save_all_histories(self):
        """保存所有训练运行的历史记录"""
        for run_name in self.all_histories:
            self.save_history(run_name)

    def update_plots(self):
        """更新并保存所有指标图表"""
        if not self.all_histories:
            return

        # 确保图表目录存在
        plot_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # 收集所有运行名称
        run_names = list(self.all_histories.keys())

        # 绘制损失曲线
        plt.figure(figsize=(12, 8))

        # 训练损失
        for run_name in run_names:
            history = pd.DataFrame(self.all_histories[run_name])
            if 'Train Loss' in history.columns:
                plt.plot(history['Epoch'], history['Train Loss'],
                         label=f'{run_name} Train Loss', linestyle='-')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "all_train_loss.png"))
        plt.close()

        # 验证损失
        plt.figure(figsize=(12, 8))
        for run_name in run_names:
            history = pd.DataFrame(self.all_histories[run_name])
            if 'Valid Loss' in history.columns:
                plt.plot(history['Epoch'], history['Valid Loss'],
                         label=f'{run_name} Valid Loss', linestyle='--')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "all_valid_loss.png"))
        plt.close()

        # 准确率曲线
        plt.figure(figsize=(12, 8))
        for run_name in run_names:
            history = pd.DataFrame(self.all_histories[run_name])
            if 'Validation Accuracy' in history.columns:
                plt.plot(history['Epoch'], history['Validation Accuracy'],
                         label=f'{run_name} Accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "all_accuracy.png"))
        plt.close()

    def finish(self):
        """完成日志记录"""
        self.save_all_histories()
        self.update_plots()


def set_optimizer(model, optimizer_choice, learning_rate, ):
    if optimizer_choice == "SGD" or optimizer_choice == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_choice == "Adam" or optimizer_choice == "ADAM" or optimizer_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return optimizer


def flatten(l):
    return [item for sublist in l for item in sublist]


def data_split(data, scale):
    train_sample = random.sample(data.tolist(), round(len(data) * scale[0]))
    remain = list(set(data).difference(set(train_sample)))
    val_per = scale[1] / (scale[1] + scale[2])
    val_sample = random.sample(remain, round(len(remain) * val_per))
    test_sample = list(set(remain).difference(set(val_sample)))
    return train_sample, val_sample, test_sample


def generate_train_valid_data(config):
    connected_csv_path = config['connected_csv_path']
    target_name = config["target_label"]
    train_radio = config.get("train_radio", 0.8)
    index_name = config["index_name"]
    try:
        data = pd.read_csv(connected_csv_path)
    except Exception as e:
        print(e)

    group = data.groupby(target_name)
    train_id = []
    valid_id = []
    scale = [train_radio, 1 - train_radio, 0]
    for key, value in group:
        train_sample, val_sample, test_sample = data_split(value[index_name], scale)
        train_id.append(train_sample)
        valid_id.append(val_sample)

    train_id = flatten(train_id)
    valid_id = flatten(valid_id)

    train_data = pd.merge(pd.DataFrame({index_name: train_id}), data, on=index_name, how="inner")
    valid_data = pd.merge(pd.DataFrame({index_name: valid_id}), data, on=index_name, how="inner")

    return train_data, valid_data


def loss_backward(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def load_config(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def main(args):
    global WANDB_AVAILABLE

    config = load_config(args.config_file)
    # 配置参数
    num_classes = config['num_classes']
    pretrain = config['model']['pretrain']
    pretrain_weight_path = config['model']['pretrain_weight_path']

    # 模型选择和初始化
    model_name = config['model']['name']
    epochs = config['epochs']
    batch_size = config['batch_size']
    train_valid_dir = config.get("train_valid_dir", None)  # 避免KeyError
    train_radio = config.get("train_radio", 0.8)

    # 初始化SimpleLogger（如果需要）
    simple_logger = None

    for i in range(args.train_time):
        # 日志初始化
        logger = None
        use_wandb = False
        run_name = f"time{i}"

        if WANDB_AVAILABLE:
            try:
                # 设置Wandb超时时间
                wandb_settings = wandb.Settings(start_method="thread", timeout=30)

                # 尝试初始化Wandb，添加超时处理
                start_time = time.time()
                wandb.init(
                    project="classification",
                    group=f"Training_{i}_runs",
                    config={
                        "model_name": model_name,
                        "pretrained": pretrain,
                        "optimizer": config['optimizer']['name'],
                        "learning_rate": config['optimizer']['learning_rate'],
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "train_radio": train_radio
                    },
                    name=run_name,
                    settings=wandb_settings
                )
                elapsed_time = time.time() - start_time

                if elapsed_time >= 30:  # 手动检查超时
                    raise TimeoutError("Wandb初始化超时")

                logger = wandb
                use_wandb = True
                print("Wandb初始化成功")

            except (wandb.errors.CommError, timeout, TimeoutError) as e:
                print(f"Wandb连接超时或通信错误: {e}")
                print("将使用SimpleLogger替代...")
                WANDB_AVAILABLE = False
            except Exception as e:
                print(f"Wandb初始化失败: {e}")
                print("将使用SimpleLogger替代...")
                WANDB_AVAILABLE = False

        if not use_wandb:
            # 初始化或获取SimpleLogger实例
            if simple_logger is None:
                log_dir = os.path.join(args.output_path, "logs")
                os.makedirs(log_dir, exist_ok=True)
                simple_logger = SimpleLogger(log_dir, config={
                    "model_name": model_name,
                    "pretrained": pretrain,
                    "optimizer": config['optimizer']['name'],
                    "learning_rate": config['optimizer']['learning_rate'],
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "train_radio": train_radio,
                    "train_time": args.train_time
                })
            # 添加新的训练运行
            simple_logger.add_run(run_name)
            logger = simple_logger

        # 输出路径管理
        time_out_path = os.path.join(args.output_path, run_name)
        os.makedirs(time_out_path, exist_ok=True)

        # 数据集加载
        if train_valid_dir:
            select_train_data = pd.read_csv(os.path.join(train_valid_dir, "train.csv"))
            select_valid_data = pd.read_csv(os.path.join(train_valid_dir, "valid.csv"))
        else:
            select_train_data, select_valid_data = generate_train_valid_data(config)
            select_train_data.to_csv(os.path.join(time_out_path, "train.csv"), encoding="utf_8_sig", index=False)
            select_valid_data.to_csv(os.path.join(time_out_path, "valid.csv"), encoding="utf_8_sig", index=False)

        # 数据加载器
        train_dataset = BaseDataset(select_train_data, config)
        valid_dataset = BaseDataset(select_valid_data, config)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # 模型初始化
        if pretrain:
            model = create_vit(model_name=model_name, n_class=num_classes,
                               pretrain=pretrain_weight_path).to(device)
        else:
            model = create_vit(model_name=model_name, n_class=num_classes).to(device)

        optimizer_name = config['optimizer']['name']
        learning_rate = config['optimizer']['learning_rate']
        optimizer_step_size = config['optimizer'].get('step_size', 10)
        optimizer_gamma = config['optimizer'].get('gamma', 0.1)
        optimizer = set_optimizer(model, optimizer_name, learning_rate)
        scheduler = StepLR(optimizer, step_size=optimizer_step_size, gamma=optimizer_gamma)
        criterion = torch.nn.CrossEntropyLoss()

        best_loss = float('inf')

        train_loss_list, valid_loss_list, acc_list = [], [], []
        for epoch in tqdm(range(epochs)):
            model.train()
            train_loss = 0.0

            for features, labels in train_loader:
                features = features.float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_loss_list.append(train_loss)

            # 验证阶段
            model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for features, labels in valid_loader:
                    features = features.float().to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            valid_loss /= len(valid_loader)
            valid_loss_list.append(valid_loss)
            accuracy = correct / total
            acc_list.append(accuracy)

            # 更新学习率
            scheduler.step()

            # 记录指标
            metrics = {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Valid Loss": valid_loss,
                "Validation Accuracy": accuracy
            }

            if use_wandb:
                logger.log(metrics)
            else:
                # 对于SimpleLogger，需要指定运行名称
                logger.log(run_name, metrics)

            # 保存最佳模型
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(time_out_path, "best_model.pth"))

        # 保存历史记录
        history = pd.DataFrame({
            "Epoch": range(epochs),
            "Train Loss": train_loss_list,
            "Valid Loss": valid_loss_list,
            "Accuracy": acc_list
        })
        history.to_csv(os.path.join(time_out_path, "training_history.csv"), index=False)

        # 结束日志记录
        if use_wandb:
            logger.finish()
        else:
            # 每次训练完成后更新图表
            simple_logger.save_history(run_name)
            simple_logger.update_plots()
            print(f"训练 {run_name} 的图表已更新")

    # 确保SimpleLogger完成所有记录
    if simple_logger is not None:
        simple_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_time', type=int, default=3, help='The number of deep learning loops')
    parser.add_argument('--output_path', type=str, default='output_path',
                        help='--output_path')
    parser.add_argument('--config_file', type=str, default=r"docs/config.yaml",
                        help='config files')
    main(parser.parse_args())