import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attetion
    """
    def forward(self, dim_head, Q, K, V):
        scores = Q.matmul(K.transpose(-1, -2)) / np.sqrt(dim_head)
        attention = F.softmax(scores, dim=-1)
        out = attention.matmul(V)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention with FIXED Dimension
    Args:
        :dim: input channel dimension
        :num_heads: number of heads of attention (default=4)
        :dim_head: dimension of Query, Key, Value (default=32)
        :device: which device to use (e.g. cuda:0)
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 dim_head: int = 32,
                 device: str = None):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.device = device

        # self.sqrt_dim = np.sqrt(dim_head)
        self.inner_dim = num_heads * dim_head

        self.to_qkv = nn.Conv2d(in_channels=self.dim, out_channels=self.inner_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(in_channels=self.inner_dim, out_channels=self.dim, kernel_size=1),
                                    nn.GroupNorm(num_groups=1, num_channels=self.dim))
        
        self.spda = ScaledDotProductAttention()

    def forward(self, x):
        batch, channel, height, width = x.size()
        qkv = self.to_qkv(x) # (b, c, h, w) -> (b, inner * 3, h, w)
        qkv = qkv.chunk(chunks=3, dim=1) # (b, inner * 3, h, w) -> (b, inner, h, w) * 3
        q, k, v = map(lambda x: x.view(batch, self.num_heads, self.dim_head, -1), qkv) # (b, inner, h, w) -> (b, num_heads, dim_head, h * w)
        
        out = self.spda(self.dim_head, q, k, v) # attention -> (b, num_heads, dim_head, h * w)
        
        out = out.view(batch, self.inner_dim, height, width) # (b, num_heads, dim_head, h * w) -> (b, num_heads * dim_head, h, w)
        out = self.to_out(out) # (b, num_heads * dim_head, h, w) -> (b, c, h, w)
        return out
    

class LinearAttention(nn.Module):
    """
    https://arxiv.org/abs/1812.01243
    Linear Attention with FIXED dimension
    Args:
        :dim: input channel dimension
        :num_heads: number of heads of attention (default=4)
        :dim_head: dimension of Query, Key, Value (default=32)
        :device: which device to use (e.g. cuda:0)
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 dim_head: int = 32,
                 device: str = None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.device = device

        self.sqrt_dim = np.sqrt(dim_head)
        self.inner_dim = num_heads * dim_head

        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=self.inner_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(in_channels=self.inner_dim, out_channels=self.dim, kernel_size=1),
                                    nn.LayerNorm(self.dim))

    def forward(self, x):
        batch, channel, height, width = x.size()
        qkv = self.to_qkv(x) # (b, c, h, w) -> (b, inner * 3, h, w)
        qkv = qkv.chunk(chunks=3, dim=1) # (b, inner * 3, h, w) -> (b, inner, h, w) * 3
        q, k, v = map(lambda x: x.view(batch, self.num_heads, self.dim_head, -1), qkv) # (b, inner, h, w) -> (b, num_heads, dim_head, h * w)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.sqrt_dim
        v = v / (height * width)

        context = k.matmul(v.transpose(-1, -2))

        out = context.transpose(-1, -2).matmul(q)
        out = out.view(batch, self.inner_dim, height, width)
        out = self.to_out(out)
        return out
    
