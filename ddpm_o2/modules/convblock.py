import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    code from https://github.com/joe-siyuan-qiao/WeightStandardization
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1,
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True):
        super(WeightStandardizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + eps
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResnetBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 *,
                 time_emb_dim = None,
                 groups = 8):
        super().__init__()

        self.weightconv1 = WeightStandardizedConv2d(in_channels=dim, out_channels=dim_out)
        self.weightconv2 = WeightStandardizedConv2d(in_channels=dim_out, out_channels=dim_out)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.actv = nn.SiLU()
        self.res_conv = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return

    pass