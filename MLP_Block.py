import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import numpy as np
import os
import copy
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.long

# reference & tutorial : http://einops.rocks/pytorch-examples.html

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


class mlp_block(nn.Module):
    def __init__(self, emb_dim: int = 16*16*3, forward_dim: int = 4, dropout_ratio: float = 0.2, **kwargs):     # dim 차원 맞춰 줘야함 -> 수정 요망
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim, forward_dim * emb_dim)   
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear_2 = nn.Linear(forward_dim * emb_dim, emb_dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x) 
        x = self.linear_2(x)
        return x