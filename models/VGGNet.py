import torch
import torch.nn as nn

dict_blocks = {'vgg11': [(3), (3, 6), (6, 11), (11, 16), (16)],
               'vgg13': [(5), (5, 10), (10, 15), (15, 20), (20)],
               'vgg16': [(5), (5, 10), (10, 17), (17, 24), (24)]}


class MyVgg(nn.Module):
    def __init__(self, model_name, num_blocks, num_classes, pretrained):
        super().__init__()
        if model_name not in dict_blocks.keys():
            raise ValueError("vgg model name not found, available models: {}".format(dict_blocks.keys()))

        total_blocks = 4
        if num_blocks > total_blocks or num_blocks < 1:
            raise ValueError("num_blocks should be an integer betwwen 1 and {} ".format(total_blocks))

        basemodel = torch.hub.load('pytorch/vision:v0.9.0', model_name, pretrained=pretrained)
        num_features = 64
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        layers = list(basemodel.children())[0]

        self.conv1 = nn.Sequential(*layers[:dict_blocks[model_name][0]])

        if self.num_blocks == 1:
            self.conv2 = nn.Sequential(*layers[dict_blocks[model_name][1][0]:dict_blocks[model_name][1][1]])
            num_features = 128

        if self.num_blocks == 2:
            self.conv2 = nn.Sequential(*layers[dict_blocks[model_name][1][0]:dict_blocks[model_name][1][1]])
            self.conv3 = nn.Sequential(*layers[dict_blocks[model_name][2][0]:dict_blocks[model_name][2][1]])
            num_features = 256

        if self.num_blocks == 3:
            self.conv2 = nn.Sequential(*layers[dict_blocks[model_name][1][0]:dict_blocks[model_name][1][1]])
            self.conv3 = nn.Sequential(*layers[dict_blocks[model_name][2][0]:dict_blocks[model_name][2][1]])
            self.conv4 = nn.Sequential(*layers[dict_blocks[model_name][3][0]:dict_blocks[model_name][3][1]])
            num_features = 512

        if self.num_blocks == 4:
            self.conv2 = nn.Sequential(*layers[dict_blocks[model_name][1][0]:dict_blocks[model_name][1][1]])
            self.conv3 = nn.Sequential(*layers[dict_blocks[model_name][2][0]:dict_blocks[model_name][2][1]])
            self.conv4 = nn.Sequential(*layers[dict_blocks[model_name][3][0]:dict_blocks[model_name][3][1]])
            self.conv5 = nn.Sequential(*layers[dict_blocks[model_name][4]:])
            num_features = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        if self.num_blocks == 1:
            x = self.conv2(x)
        if self.num_blocks == 2:
            x = self.conv2(x)
            x = self.conv3(x)
        if self.num_blocks == 3:
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
        if self.num_blocks == 4:
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def VggNet(model_name, num_blocks, num_classes=1, pretrained=True):
    model = MyVgg(model_name, num_blocks, num_classes, pretrained)

    return model

