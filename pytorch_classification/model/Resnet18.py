import timm
import torch
import torch.nn as nn
from torchvision import models


def resnet18(n_class=2, pretrain=""):
    model = models.resnet18()
    try:
        weights = torch.load(pretrain)
        model.load_state_dict(weights)
    except:
        pass
    model.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)

    return model


if __name__ == "__main__":
    model = resnet18(n_class=21)
    print(model)
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    print(out.shape)
    print("param #", sum(p.numel() for p in model.parameters()))