import os.path
import random
import torch
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, data, config):
        try:
            if not isinstance(config, dict):
                with open(config, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
        except Exception as e:
            print(f"fail load config:{e}")

        try:
            data = pd.read_csv(config['connected_csv_path'])
        except Exception as e:
            print(e)
        self.image_path = data[config["index_name"]]
        self.target_name = config["target_label"]
        self.label_dict = {value: key for key, value in config["label"].items()}
        self.label = data[self.target_name].map(self.label_dict)

        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ])
        # self.base_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=(0.48422232, 0.49004802, 0.4504976),
        #         std=(0.21796763, 0.20201482, 0.19578394)),
        # ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        base_tensor = self.base_transform(image)
        return base_tensor, self.label[idx]


if __name__ == '__main__':
    pass
    # data = PrepareData("D:/pycharm/MSI/Data/485_raw/485/train", "D:/pycharm/MSI/Data/485_raw/485/val")
    # # print(len(data.val_data))
    # # print(data.val_data[0])
    # # print(data.val_data[:32][0])
    # for feature, label, lens in data.split_batch(data.train_data, 32):
    #     print(feature.shape, label, lens)
    # import h5py

    # feature = h5py.File(os.path.join("D:/pycharm/MSI/Data/1221_raw_all", 'MP2019-08728.h5'), "r")
    # feat = feature["feats"][:]
    # print(feat.shape)
    # feature.close()
    # for it in data.val_data:
    #     print(it[0].shape, it[1])
    # data = MILdataset("C:/MSI/CRC_FUSCC_Features", mode="test")
    # train_dataloader = DataLoader(data, batch_size=1, shuffle=False)
    # # for it in data:
    # #     print(it[0].shape, it[1])
    # # print(train_dataloader)
    # for it in train_dataloader:
    #     print(it[0].shape, it[1])
