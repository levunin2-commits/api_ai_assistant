import torch
import torch.nn as nn
from torchvision import models


class MyMobileNetV2(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.net.classifier[1] = nn.Linear(self.net.last_channel, num_classes)
        # fc_inputs = self.net.classifier[1].in_features
        # self.net.classifier = nn.Sequential(
        #     nn.Linear(fc_inputs, num_classes),
        # )

    def forward(self, x):
        return self.net(x)
