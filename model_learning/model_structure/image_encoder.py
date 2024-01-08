import torch.nn as nn
from torchvision import models
import torch


class ImageEncoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(self.resnet50.fc.out_features, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc1(x)
        return x


class ImageEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(self.resnet50.fc.out_features, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 128)
        self.fc3 = nn.Linear(self.fc2.out_features, 64)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ImageEncoderV3(nn.Module):
    def __init__(self, dropout_prob, embedding_size):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.fc = nn.Linear(self.resnet50.fc.out_features, embedding_size)

        torch.nn.init.kaiming_normal_(self.fc.weight)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
