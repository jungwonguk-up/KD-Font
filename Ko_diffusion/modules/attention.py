import torch
import torch.nn.functional as F
from torch import nn, einsum


class Attention(nn.Module):
    def __init__(self,
                 query_dim: int,
                 context_dim: int = None,
                 num_heads: int = 4,
                 head_dim: int = None,
                 dropout: float = 0.):
        super().__init__()
        """
        Attention Block for Self-Attention and Cross-Attention
        :query_dim: Dimension of query
        :context_dim: Dimension of context, None parameter works as Self-Attention (default=None)
        :num_heads: Number of Head (default=4)
        :head_dim: Dimension of Head, None parameter define head_dim to query_dim (default=None)
        :dropout: DropOut Rate (default=0.)
        """
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




