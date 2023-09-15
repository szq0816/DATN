import torch.nn.functional as F
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SEBlock(nn.Module):
    def __init__(self, in_channel, patch_size, mid_dim):
        super(SEBlock, self).__init__()
        self.GAPool = nn.AvgPool2d(patch_size, stride=1)
        self.fc_reduction = nn.Linear(in_features=in_channel, out_features=in_channel // mid_dim)
        self.fc_extention = nn.Linear(in_features=in_channel // mid_dim, out_features=in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se_out = self.GAPool(x)
        se_out = se_out.view(se_out.size(0), -1)
        se_out = F.relu(self.fc_reduction(se_out), inplace=True)
        se_out = self.fc_extention(se_out)
        se_out = self.sigmoid(se_out)
        se_out = se_out.view(se_out.size(0), se_out.size(1), 1, 1)  # batch_size x channel x 1 x 1
        return se_out * x

class localBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(localBlock, self).__init__()
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv',
                              nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, padding=0,
                                        bias=False))
        self.conv0.add_module('bn',
                              nn.BatchNorm2d(256))
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv',
                              nn.Conv3d(1, 12, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1),
                                        bias=False))
        self.conv1.add_module('bn', nn.BatchNorm3d(12))
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv',
                              nn.Conv3d(12, 12, kernel_size=(1, 1, 7), stride=1, padding=(0, 0, 3),
                                        bias=False))
        self.conv2.add_module('bn',
                              nn.BatchNorm3d(12))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv',
                              nn.Conv3d(12, 1, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1),
                                        bias=False))
        self.conv3.add_module('bn',
                              nn.BatchNorm3d(1))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv',
                              nn.Conv2d(256, out_channel, kernel_size=1, stride=1, padding=0,
                                        bias=False))
        self.conv4.add_module('bn',
                              nn.BatchNorm2d(out_channel))

    def forward(self, x):
        x = F.relu(self.conv0.forward(x), inplace=True)
        x1 = x.unsqueeze(1)
        x1 = x1.transpose(-1, 2)
        x1 = F.relu(self.conv1.forward(x1), inplace=True)
        x1 = F.relu(self.conv2.forward(x1), inplace=True)
        x1 = F.relu(self.conv3.forward(x1), inplace=True)
        x1 = x1.transpose(-1, 2)
        x1 = x1.squeeze(1)
        x = F.relu(self.conv4.forward(x1), inplace=True)
        return x