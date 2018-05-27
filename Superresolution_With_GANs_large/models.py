# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802
"""

import torch.nn as nn
import torchvision.models as models


class resnet_generator(nn.Module):
    def __init__(self):
        super(resnet_generator, self).__init__()

        resnet18 = models.resnet18(pretrained=True)

        for param in resnet18.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), bias=False)
        nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.layer1 = resnet18.layer1

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv2.weight)
        self.bb1 = resnet18.layer2[1]

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv3.weight)
        self.bb2 = resnet18.layer3[1]

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv4.weight)
        self.bb3 = resnet18.layer4[1]

        self.conv5 = nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.xavier_normal(self.conv5.weight)
        self.bn2 = nn.BatchNorm2d(64)

        self.upsampler = upsampleBlock_modified(64, 256)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        nn.init.xavier_normal(self.conv6.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        y = x.clone()
        x = self.layer1(x)
        x = self.bb1(self.conv2(x))
        x = self.bb2(self.conv3(x))
        x = self.bb3(self.conv4(x))
        x = self.bn2(self.conv5(x)) + y
        x = self.upsampler(x)
        return self.conv6(x)


class resnet_generator_extended(nn.Module):
    def __init__(self):
        super(resnet_generator_extended, self).__init__()

        resnet = models.resnet34(pretrained=True)

        # for param in resnet.parameters():
        #    param.requires_grad = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), bias=False)
        nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv2.weight)
        self.bb1 = resnet.layer2[1]
        self.bb2 = resnet.layer2[2]
        self.bb3 = resnet.layer2[3]

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv3.weight)
        self.bb4 = resnet.layer3[1]
        self.bb5 = resnet.layer3[2]
        self.bb6 = resnet.layer3[3]
        self.bb7 = resnet.layer3[4]
        self.bb8 = resnet.layer3[5]

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.xavier_normal(self.conv4.weight)
        self.bb9 = resnet.layer4[1]
        self.bb10 = resnet.layer4[2]

        self.conv5 = nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.xavier_normal(self.conv5.weight)
        self.bn2 = nn.BatchNorm2d(64)

        self.upsampler = upsampleBlock_modified(64, 256)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        nn.init.xavier_normal(self.conv6.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        y = x.clone()
        x = self.layer1(x)
        x = self.bb3(self.bb2(self.bb1(self.conv2(x))))
        x = self.bb8(self.bb7(self.bb6(self.bb5(self.bb4(self.conv3(x))))))
        x = self.bb10(self.bb9(self.conv4(x)))
        x = self.bn2(self.conv5(x)) + y
        x = self.upsampler(x)
        return self.conv6(x)


class upsampleBlock_modified(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock_modified, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.conv.weight)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))


class DiscriminatorPretrained(nn.Module):
    def __init__(self, imgSize):
        super(DiscriminatorPretrained, self).__init__()
        self.vgg = models.vgg13_bn(pretrained=True)

        for param in self.vgg.parameters():
            param.requires_grad = False

        del self.vgg.classifier

        self.fc1 = nn.Linear(int(imgSize * imgSize /2), 1024)
        nn.init.xavier_normal(self.fc1.weight)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1)
        nn.init.xavier_normal(self.fc2.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(self.leaky(self.fc1(x)))
        return self.sigmoid(x)
