import torch
import torch.nn.functional as F
from torch import nn, einsum


class GEGLU(nn.Module):
    """
    https://arxiv.org/abs/2002.05202v1
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out*2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, diim=-1)
        return x * F.gelu(gate)


class Attention(nn.Module):
    """
    Attention Block for Self-Attention and Cross-Attention.
    :query_dim: Dimension of query
    :context_dim: Dimension of context, None parameter works as Self-Attention (default=None)
    :num_heads: Number of Head (default=4)
    :head_dim: Dimension of Head, None parameter define head_dim to query_dim (default=None)
    :dropout: DropOut Rate (default=0.)
    """
    def __init__(self,
                 query_dim: int,
                 context_dim: int = None,
                 num_heads: int = 4,
                 head_dim: int = None,
                 dropout: float = 0.
                 ):
        super().__init__()
        assert query_dim 

        if not head_dim:
            head_dim = query_dim
        inner_dim = num_heads * head_dim

        self.cross = True if context_dim is not None else False
        if not self.cross:
            context_dim = query_dim
        
        self.scale = head_dim ** -0.5
        self.heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(p=dropout)
        )
    
    def forword(self, x, context=None):
        if context is not None and self.cross == False:
            raise ValueError("context should None if context_dim is None")

        h = self.heads
        d = self.head_dim
        b = x.size()[0]

        q = self.to_q(x)
        context = context if context is not None else x # if context is None, work self attn
        k = self.to(context)
        v = self.to(context)

        q, k, v = map(lambda t: t.view(b, -1, h, d).permute(0, 2, 1, 3).contiguous().view(b*h, -1, d), (q, k, v)) # b n (h d) -> (b h) n d

        sim = einsum('b i d, b j d -> b i j' , q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out.view(b, h, -1, d).permute(0, 2, 1, 3).contiguous().view(b, -1, h*d) # (b h) n d -> b n (h d)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    Basic Trasformer Block include self-attention, cross-attention, feedforward network.
    :dim: Demension of input tensor
    :num_heads: Number of Attention Head 
    :head_dim: Dimension of Attention Head
    :context_dim: Dimension of context
    :ff_dim_mult: Multiply Value of Feed Forward Network Inner Dimension (default=4)
    :use_GEGLU: (default=True)
    :dropout: DropOut Rate (default=0.)
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 context_dim: int = None,
                 ff_dim_mult: int = 4,
                 use_GEGLU: bool = True,
                 dropout: float = 0.,
                 ):
        super().__init__()
        ff_inner_dim = int(dim * ff_dim_mult)
        ff_layer = [nn.Linear(dim, ff_inner_dim), nn.GELU()] if not use_GEGLU else [GEGLU(dim, ff_inner_dim)]

        self.attn1 = Attention(query_dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout) # self-attention
        self.attn2 = Attention(query_dim=dim, context_dim=context_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout) # cross-attention

        self.ff = nn.Sequential(
            *ff_layer,
            nn.Dropout(p=dropout),
            nn.Linear(ff_inner_dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTrasformerBlock(nn.Module):
    """
    Trasformer block for image-like data.
    """
    def __init__(self, in_channels, num_head, head_dim, depth, dropout, context_dim):
        super().__init__()

    def forward(self, x, context=None):
        return x
    

class TrasformerBlock(nn.Module):
    """
    Trasformer Block for Unet.
    :in_channels: Number of input tensor channels 
    :num_heads: Number of Attention Head 
    :head_dim: Dimension of Attention Head
    :context_dim: Dimension of context
    :depth: Number of Transformer block Depth (default=1)
    :dropout: DropOut Rate (default=0.)
    :use_spatial: use Spatial Trasformer Block 
    """
    def __init__(self, 
                 in_channels: int, 
                 num_heads: int, 
                 head_dim: int, 
                 context_dim: int = None,
                 depth: int = 1, 
                 dropout: float = 0., 
                 use_spatial: bool = False,
                 ):
        super().__init__()
        self.use_spatial = use_spatial
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.transformer_block = nn.ModuleList([])
        if use_spatial: # not use currently 
            self.trasformer_block.append(SpatialTrasformerBlock(in_channels, num_heads, head_dim, depth, dropout, context_dim))
        else:
            for _ in range(depth):
                self.transformer_block.append(BasicTransformerBlock(in_channels, num_heads, head_dim, dropout, context_dim))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x) if self.use_spatial else x
        x = x.permute(0, 2, 3, 1).view(b, -1, c) # b c h w -> b (h w) c
        for block in self.transformer_block:
            x = block(x, context=context)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2) # b (h w) c -> b c h w
        x = x + x_in if self.use_spatial else x
        return x
