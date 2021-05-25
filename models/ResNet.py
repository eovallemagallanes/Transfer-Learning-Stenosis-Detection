import torch
import torch.nn as nn

resnet_models = ['resnet18', 'resnet34', 'resnet50']


class MyResNet(nn.Module):
    def __init__(self, model_name, num_blocks, num_classes, pretrained):
        super().__init__()
        if model_name not in resnet_models:
            raise ValueError("resnet model name not found, available models: {}".format(resnet_models))

        total_blocks = 4
        if num_blocks > total_blocks or num_blocks < 1:
            raise ValueError("num_blocks should be an integer betwwen 1 and {} ".format(total_blocks))

        basemodel = torch.hub.load('pytorch/vision:v0.9.0', model_name, pretrained=pretrained)
        num_features = 64
        self.inplanes = 64
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.conv1 = basemodel.conv1
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.num_blocks == 1:
            self.layer1 = basemodel.layer1
            num_features = 64

        if self.num_blocks == 2:
            self.layer1 = basemodel.layer1
            self.layer2 = basemodel.layer2
            num_features = 128

        if self.num_blocks == 3:
            self.layer1 = basemodel.layer1
            self.layer2 = basemodel.layer2
            self.layer3 = basemodel.layer3
            num_features = 256

        if self.num_blocks == 4:
            self.layer1 = basemodel.layer1
            self.layer2 = basemodel.layer2
            self.layer3 = basemodel.layer3
            self.layer4 = basemodel.layer4
            num_features = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.num_blocks == 1:
            x = self.layer1(x)
        if self.num_blocks == 2:
            x = self.layer1(x)
            x = self.layer2(x)
        if self.num_blocks == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        if self.num_blocks == 4:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResidualNet(model_name, num_blocks, num_classes=1, pretrained=True):
    model = MyResNet(model_name, num_blocks, num_classes, pretrained)

    return model
