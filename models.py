import torch.nn as nn
from torchvision import models

def get_model(name: str, num_classes: int):
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model: {name}")