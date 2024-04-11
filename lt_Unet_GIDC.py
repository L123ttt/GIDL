# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
    定义模型
        模型结构
        权重初值
        激活函数
        DGI算法
        。。。
"""


def hard_sigmoid(x):
    return torch.clamp((x + 1.) / 2., 0, 1)


def round_through(x):
    rounded = torch.round(x)
    return x + rounded - x.detach()


def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))


def binarization(W, H=1, binary=True, deterministic=False, stochastic=False):
    if not binary or (deterministic and stochastic):
        Wb = W
    else:
        Wb = H * binary_sigmoid_unit(W / H)  # 使用binary_sigmoid_unit函数
    return Wb


def DGI_reconstruction(y, patterns, num_patterns, img_W, img_H, g_factor=0.5):
    # Reshape y
    y = y.view(num_patterns, 1)
    # Reshape patterns and transpose
    patterns = patterns.reshape(num_patterns, img_W * img_H)
    # Calculate comput1
    mean_patterns = patterns.mean(0).unsqueeze(0)
    ones = torch.ones(num_patterns, 1, device=y.device)
    comput1 = patterns - ones.mm(mean_patterns)
    # Calculate comput2
    comput2 = patterns.sum(1)
    # Calculate gamma
    gamma = g_factor * y.mean() / (comput2.mean() + 1e-8)
    temp = gamma * comput2.view(num_patterns, 1)
    # Calculate DGI
    DGI = (y - temp).t().mm(comput1)  # Transpose (y - temp)
    # Normalize DGI
    DGI = (DGI - DGI.min()) / (DGI.max() - DGI.min() + 1e-8)
    # Reshape and transpose DGI
    DGI = DGI.view(1, img_W, img_H, 1).permute(0, 3, 1, 2)
    return DGI


def image_cut_by_std(img, pram):
    tmax = np.mean(img) + pram*np.std(img)
    tmin = np.mean(img) - pram*np.std(img)
    img[img>tmax] = tmax
    img[img<tmin] = tmin
    return img


class Net(nn.Module):
    def __init__(self, img_W, img_H, num_patterns):
        super(Net, self).__init__()

        self.patterns = nn.Parameter(torch.randn(1, num_patterns, img_W, img_H))

        self.conv1 = nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv8 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, batch_size, num_patterns, patterns, img_W, img_H):
        x = x.expand(-1, num_patterns, -1, -1)  # 在通道维度上扩展为 num_patterns 个通道
        # 逐元素相乘
        multiplied = patterns * x
        y = multiplied.sum(dim=(2, 3)).view(batch_size, num_patterns, 1, 1)
        y = (y - y.mean()) / (y.std() + 1e-8)
        y_temp = y[:1, :, :, :]
        DGI_R = DGI_reconstruction(y_temp, self.patterns, num_patterns, x.size(2), x.size(3), 0.5)
        for i in range(1, x.size(0)):
            y_i = y[i:i + 1, :, :, :]
            DGI_temp = DGI_reconstruction(y_i, self.patterns, num_patterns, x.size(2), x.size(3), 0.5)
            DGI_R = torch.cat([DGI_R, DGI_temp], dim=0)
        DGI_R = (DGI_R - DGI_R.min()) / (DGI_R.max() - DGI_R.min())

        temp = torch.reshape(DGI_R, (batch_size, 1, img_W, img_H))

        conv1 = F.leaky_relu(self.conv1(temp))
        conv1_1 = F.leaky_relu(self.conv1_1(conv1))
        Maxpool_1 = F.max_pool2d(conv1_1, kernel_size=2, stride=2)

        conv2 = F.leaky_relu(self.conv2(Maxpool_1))
        conv2_1 = F.leaky_relu(self.conv2_1(conv2))
        Maxpool_2 = F.max_pool2d(conv2_1, kernel_size=2, stride=2)

        conv3 = F.leaky_relu(self.conv3(Maxpool_2))
        conv3_1 = F.leaky_relu(self.conv3_1(conv3))
        Maxpool_3 = F.max_pool2d(conv3_1, kernel_size=2, stride=2)

        conv4 = F.leaky_relu(self.conv4(Maxpool_3))
        conv4_1 = F.leaky_relu(self.conv4_1(conv4))
        Maxpool_4 = F.max_pool2d(conv4_1, kernel_size=2, stride=2)

        conv5 = F.leaky_relu(self.conv5(Maxpool_4))
        conv5_1 = F.leaky_relu(self.conv5_1(conv5))

        conv6 = F.leaky_relu(self.conv6(conv5_1))
        merge1 = torch.cat([conv4_1, conv6], dim=1)
        conv6_1 = F.leaky_relu(self.conv6_1(merge1))
        conv6_2 = F.leaky_relu(self.conv6_2(conv6_1))

        conv7 = F.leaky_relu(self.conv7(conv6_2))
        merge2 = torch.cat([conv3_1, conv7], dim=1)
        conv7_1 = F.leaky_relu(self.conv7_1(merge2))
        conv7_2 = F.leaky_relu(self.conv7_2(conv7_1))

        conv8 = F.leaky_relu(self.conv8(conv7_2))
        merge3 = torch.cat([conv2_1, conv8], dim=1)
        conv8_1 = F.leaky_relu(self.conv8_1(merge3))
        conv8_2 = F.leaky_relu(self.conv8_2(conv8_1))

        conv9 = F.leaky_relu(self.conv9(conv8_2))
        merge4 = torch.cat([conv1_1, conv9], dim=1)
        conv9_1 = F.leaky_relu(self.conv9_1(merge4))
        conv9_2 = F.leaky_relu(self.conv9_2(conv9_1))

        conv10 = F.leaky_relu(self.conv10(conv9_2))
        conv10 = (conv10 - conv10.min()) / (conv10.max() - conv10.min())

        return DGI_R, patterns, conv10


def create_model(img_W, img_H, num_patterns):
    net = Net(img_W, img_H, num_patterns)
    return net
