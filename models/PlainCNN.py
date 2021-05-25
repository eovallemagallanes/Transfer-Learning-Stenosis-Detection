"""
Plain CNN implementation for stenosis detection
Based on keras implementation: https://github.com/KarolAntczak/DeepStenosisDetection
"""


import torch
import torch.nn as nn


class Baseline1(nn.Module):

    def __init__(self, in_channels=1):
        super(Baseline1, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 8, kernel_size=7, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=7, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=7, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.dropout2 = nn.Dropout(p=0.5)
        self.conv4 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=7, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 16)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)

        return x