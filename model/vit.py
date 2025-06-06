import timm
import torch
import torch.nn as nn
from torchvision import models


def create_vit(model_name="vit_small_patch16_224", n_class=2, pretrain=""):
    model = timm.create_model(model_name=model_name, checkpoint_path=pretrain)
    model.head = nn.Linear(in_features=384, out_features=n_class, bias=True)

    return model


if __name__ == "__main__":
    path = r"D:\Deeplearning_weights\VIT\S_16-i21k-imagenet2012_224.npz"
    model = timm.create_model(model_name="vit_small_patch16_224", checkpoint_path=path)

    print(model.default_cfg)
    # print(model.state_dict())
    # torch.seed(0)
    # x = torch.rand(1, 3, 224, 224)
    # out = model(x)
    # print(out[0])
    # # print("param #", sum(p.numel() for p in model.parameters()))