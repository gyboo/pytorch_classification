import os

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


def save_data_aug(image, name, output_path):
    hor_img = transforms.RandomHorizontalFlip(p=1)(image)  # 随机水平翻转, p为概率
    hor_img.save(os.path.join(output_path, name + "_Horizontal.tif"))
    ver_img = transforms.RandomVerticalFlip(p=1)(image)  # 随机垂直翻转，p为概率
    ver_img.save(os.path.join(output_path, name + "_Vertical.tif"))
    # ot_img = transforms.RandomRotation(90)(image)  # 随机在（-15， 15）度旋转
    ot_img = image.rotate(90)
    ot_img.save(os.path.join(output_path, name + "_Rotation90.tif"))


if __name__ == "__main__":
    data_path = r"D:\Dataset\UCMerced_LandUse"
    table_path = r"D:\pycharm\遥感图像分类\docs\train.csv"
    data = pd.read_csv(table_path)
    out_path = r"D:\Dataset\UCMerced_LandUse_Augmentation"
    os.makedirs(out_path, exist_ok=True)
    name_list, label_list = [], []
    for name in tqdm(data["image_name"].values):
        image = Image.open(os.path.join(data_path, name + ".tif"))
        image.save(os.path.join(out_path, name + ".tif"))
        save_data_aug(image, name, out_path)
        label = data.loc[data["image_name"] == name, "ground_truth"].values[0]
        name_list.append(name)
        name_list.append(name + "_Horizontal")
        name_list.append(name + "_Vertical")
        name_list.append(name + "_Rotation90")
        for i in range(4):
            label_list.append(label)
    aug_data = pd.DataFrame({
        "image_name": name_list,
        "ground_truth": label_list
    })
    # aug_data = pd.concat([aug_data, data])
    aug_data.to_csv(r"D:\pycharm\遥感图像分类\docs\train_aug.csv", index=False, encoding="utf_8_sig")