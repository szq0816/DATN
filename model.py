import torch
import torch.nn as nn
import math
from typing import Tuple
import torch.nn.functional as F
from utils import *
from transformer import *

class DATN(nn.Module):
    def __init__(self, in_channel, num_classes, patch_size):
        super().__init__()
        channels = [64, 256]
        self.patch_size = patch_size
        self.conv_conv1 = nn.Conv2d(in_channel, channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn1 = nn.BatchNorm2d(channels[0])
        self.conv_conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn2 = nn.BatchNorm2d(channels[1])
        self.conv_conv3 = nn.Conv2d(channels[1], channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn3 = nn.BatchNorm2d(channels[0])
        self.conv_conv4 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn4 = nn.BatchNorm2d(channels[1])
        self.pool = GlobalAvgPool2d()
        self.fc = nn.Linear(channels[1], num_classes, bias=False)
        self.dropout = nn.Dropout(0.)

        self.se1 = SEBlock(channels[0], self.patch_size, 16)
        self.se2 = SEBlock(channels[1], self.patch_size, 16)
        self.se3 = SEBlock(channels[0], self.patch_size, 16)
        self.se4 = SEBlock(channels[1], self.patch_size, 16)

        self.local1 = localBlock(in_channel, channels[1])
        self.local2 = localBlock(channels[1], channels[1])

        self.spectral_attention = SpectralBlock(channels[0], 1)
        self.spatial_attention = SpatialBlock(channels[0], 4)
        self.spectral_attention1 = SpectralBlock(channels[1], 1)
        self.spatial_attention1 = SpatialBlock(channels[1], 8)
        self.spectral_attention2 = SpectralBlock(channels[0], 1)
        self.spatial_attention2 = SpatialBlock(channels[0], 4)
        self.spectral_attention3 = SpectralBlock(channels[1], 1)
        self.spatial_attention3 = SpatialBlock(channels[1], 8)


    def forward_features(self, x):
        B, _, H, W = x.shape
        x_in = x
        local1 = self.local1(x_in)
        x = self.conv_conv1.forward(x)
        x = F.relu(self.bn_bn1.forward(x), inplace=True)
        x = self.se1(x)
        x = x.flatten(2).permute(0, 2, 1)
        x, size = self.spectral_attention(x, (self.patch_size, self.patch_size))
        x, size = self.spatial_attention(x, (self.patch_size, self.patch_size))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.conv_conv2.forward(x)
        x = F.relu(self.bn_bn2.forward(x), inplace=True)
        x = self.se2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x, size = self.spectral_attention1(x, (self.patch_size, self.patch_size))
        x, size = self.spatial_attention1(x, (self.patch_size, self.patch_size))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = F.relu(x + local1, inplace=True)
        x_mid = x
        local2 = self.local2(x_mid)
        x = self.conv_conv3.forward(x)
        x = F.relu(self.bn_bn3.forward(x), inplace=True)
        x = self.se3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x, size = self.spectral_attention2(x, (self.patch_size, self.patch_size))
        x, size = self.spatial_attention2(x, (self.patch_size, self.patch_size))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.conv_conv4.forward(x)
        x = F.relu(self.bn_bn4.forward(x), inplace=True)
        x = self.se4(x)
        x = x.flatten(2).permute(0, 2, 1)
        x, size = self.spectral_attention3(x, (self.patch_size, self.patch_size))
        x, size = self.spatial_attention3(x, (self.patch_size, self.patch_size))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = F.relu(x, inplace=True)
        return F.relu(local2 + x, inplace=True)


    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(self.dropout(x)).view(-1, x.shape[1])
        x = self.fc(x)
        return x




if __name__ == '__main__':
    net = DATN(in_channel = 200,num_classes=16, patch_size= 11)
    net.eval()
    print(net)
    input = torch.randn(1, 200, 11, 11)
    y = net(input)
    print(y.shape, count_parameters(net))

