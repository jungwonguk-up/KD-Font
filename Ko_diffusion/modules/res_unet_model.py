import torch
import torch.nn.functional as F
import torch.nn as nn

from .style_encoder import style_enc_builder
from .condition import Korean_StrokeEmbedding
from .attention import TrasformerBlock


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class BlockSequential(nn.Sequential):

    def forward(self, x, t, context=None):
        for layer in self:
            if isinstance(layer, BasicBlock):
                x = layer(x, t, context)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    """
    An upsampling layer with convolution.
    """
    def __init__(self, in_channels, out_channals, padding=1):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channals

        self.conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.in_ch

        x = F.interpolate(x, scale_factor=2, mode="nearest") # mode = nearst, bilinear, ...
        x = self.conv(x)
        return x
    

class DownSample(nn.Module):
    """
    An downsampling layer with convolution.
    """
    def __init__(self, in_channels, out_channels, padding=1) :
        super().__init__()
        
        self.in_ch = in_channels
        self.out_ch = out_channels

        stride = 2
        self.conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.in_ch
        return self.conv(x)


class ResBlock(nn.Module):
    """
    A residual block
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 emb_channels: int = 256,
                 norm_num_groups: int = 8,
                 dropout: float = 0.,
                 ):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.emb_ch = emb_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.in_ch),
            nn.SELU(),
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_ch, self.out_ch)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_ch),
            nn.SELU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)) # zero_module 적용
        )

        # skip connection
        if self.out_ch == self.in_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1)

    def forward(self, x, t):
        out = self.in_layers(x)
        # t = self.emb_layer(t).type(h.type) # original code
        # print("x :", x.shape)
        # print("t :", t.shape)
        t = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        out = out + t
        out = self.out_layers(out)
        return out + self.residual(x)
    

class BasicBlock(nn.Module):
    """
    Basic Block for Unet. ResBlock + TransfomerBlock 
    :mid_block: set value True when use BasicBlock as MidBlock in UNet
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 emb_channels: int = 256,
                 num_heads: int = 4,
                 head_dim: int = 32,
                 context_dim: int = None,
                 depth: int = 1,
                 use_transformer_block: bool = True,                
                 mid_block: bool = False,
                 norm_num_groups: int = 8,
                 dropout: float = 0.):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        # self.mid_ch = in_channels * 2

        self.use_tf = use_transformer_block
        self.mid = mid_block

        if self.mid:
            assert self.in_ch == self.out_ch
            # self.mid_head_dim = self.mid_ch // num_heads

            self.res1 = ResBlock(self.in_ch, self.out_ch, emb_channels, norm_num_groups, dropout)
            self.tf = TrasformerBlock(self.out_ch, num_heads, head_dim, context_dim, depth, norm_num_groups, dropout)
            self.res2 = ResBlock(self.out_ch, self.out_ch, emb_channels, norm_num_groups, dropout)
        else:
            self.res1 = ResBlock(self.in_ch, self.out_ch, emb_channels, norm_num_groups, dropout)
            self.tf = TrasformerBlock(self.out_ch, num_heads, head_dim, context_dim, depth, norm_num_groups, dropout) if self.use_tf else None

    def forward(self, x, t, context=None):
        if (self.use_tf is False) and (self.mid is True): # mid 일 때는 항상 transformer 사용
            raise ValueError(" mid 일 때는 항상 transformer 사용!")
        
        if self.mid:
            x = self.res1(x, t)
            x = self.tf(x, context)
            x = self.res2(x, t)
        else:
            x = self.res1(x, t)
            if self.use_tf:
                x = self.tf(x, context)

        return x


class Unet(nn.Module):
    """
    UNet model with attention and timestep embedding.
    """
    def __init__(self,
                 image_size: int = 64,
                 in_channels: int = 1,
                 model_channels: int = 64,
                 out_channels: int = 1,
                 num_basic_block: int = 1,
                #  attention_resolutions: list = [4, 2, 1], # not use maybe?
                 dropout: float = 0.,
                 channel_mult: list = [1, 2, 4, 4],
                 num_heads: int = 4,
                 transformer_depth: int = 1,
                 norm_num_groups: int = 8,
                 context_dim: int = None,
                 device: str = "cuda"):
        super().__init__()
        assert context_dim

        self.img_size = image_size
        self.in_ch = in_channels
        self.model_ch = model_channels
        self.out_ch = out_channels
        self.num_block = num_basic_block
        # self.attn_res = attention_resolutions
        self.dropout = dropout
        self.ch_mult = channel_mult
        self.num_heads = num_heads
        self.depth = transformer_depth
        self.norm_num_groups = norm_num_groups
        self.context_dim = context_dim
        self.device = device

        self.enc_layer_ch_list = [i * self.model_ch for i in channel_mult]

        self.input = nn.Conv2d(self.in_ch, self.model_ch, kernel_size=3, padding=1)
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=self.norm_num_groups, num_channels=self.model_ch),
            nn.SiLU(),
            nn.Conv2d(self.model_ch, self.out_ch, kernel_size=1) # 맞나?
        )


        # self.sty_avgpool = nn.AvgPool2d(kernel_size = 16)   # batch, 128, 1


        # self.label_linear = nn.Sequential(
        #     nn.Linear(100, context_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(context_dim),
        # )

        self.cond_emb_linear = nn.Sequential(
            nn.Linear(256, self.context_dim*2),
            nn.SiLU(),
            nn.Linear(self.context_dim*2, self.context_dim)
        )

        # self.sty_linear = nn.Sequential(
        #     nn.Linear(128, self.context_dim*2),
        #     nn.SiLU(),
        #     nn.Linear(self.context_dim*2, self.context_dim),
        # )

        # self.content_linear = nn.Sequential(
        #     nn.Linear(60, self.context_dim),
        #     nn.LayerNorm(self.context_dim),
        #     nn.SiLU()
        # )

        # self.stroke_linear = nn.Sequential(
        #     nn.Linear(68, self.context_dim),
        #     nn.LayerNorm(self.context_dim),
        #     nn.SiLU()
        # )


        """임시"""
        # level1
        self.enc1 = BasicBlock(self.model_ch, self.model_ch, num_heads=self.num_heads, head_dim=(self.model_ch)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.enc2 = BasicBlock(self.model_ch, self.model_ch, num_heads=self.num_heads, head_dim=(self.model_ch)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        # level2
        self.down1 = DownSample(self.model_ch, self.model_ch)
        self.enc3 = BasicBlock(self.model_ch, self.model_ch*2, num_heads=self.num_heads, head_dim=(self.model_ch*2)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.enc4 = BasicBlock(self.model_ch*2, self.model_ch*2, num_heads=self.num_heads, head_dim=(self.model_ch*2)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        # level3
        self.down2 = DownSample(self.model_ch*2, self.model_ch*2)
        self.enc5 = BasicBlock(self.model_ch*2, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.enc6 = BasicBlock(self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        # level4
        self.down3 = DownSample(self.model_ch*4, self.model_ch*4)
        self.enc7 = BasicBlock(self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, use_transformer_block=False, norm_num_groups=self.norm_num_groups)
        self.enc8 = BasicBlock(self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, use_transformer_block=False, norm_num_groups=self.norm_num_groups)
        
        # mid
        # inner dim = in_ch * 2 
        self.mid = BasicBlock(self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, mid_block=True, norm_num_groups=self.norm_num_groups)

        # level4
        # in_channels = x_ch + skip_ch
        self.dec1 = BasicBlock(self.model_ch*4 + self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, use_transformer_block=False, norm_num_groups=self.norm_num_groups)
        self.dec2 = BasicBlock(self.model_ch*4 + self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, use_transformer_block=False, norm_num_groups=self.norm_num_groups)
        self.dec3 = BasicBlock(self.model_ch*4 + self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, use_transformer_block=False, norm_num_groups=self.norm_num_groups)
        self.up1 = UpSample(self.model_ch*4, self.model_ch*4)
        # level3
        self.dec4 = BasicBlock(self.model_ch*4 + self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.dec5 = BasicBlock(self.model_ch*4 + self.model_ch*4, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.dec6 = BasicBlock(self.model_ch*4 + self.model_ch*2, self.model_ch*4, num_heads=self.num_heads, head_dim=(self.model_ch*4)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.up2 = UpSample(self.model_ch*4, self.model_ch*4)
        # level2
        self.dec7 = BasicBlock(self.model_ch*4 + self.model_ch*2, self.model_ch*2, num_heads=self.num_heads, head_dim=(self.model_ch*2)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.dec8 = BasicBlock(self.model_ch*2 + self.model_ch*2, self.model_ch*2, num_heads=self.num_heads, head_dim=(self.model_ch*2)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.dec9 = BasicBlock(self.model_ch*2 + self.model_ch, self.model_ch*2, num_heads=self.num_heads, head_dim=(self.model_ch*2)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.up3 = UpSample(self.model_ch*2, self.model_ch*2)
        # level1
        self.dec10 = BasicBlock(self.model_ch*2 + self.model_ch, self.model_ch, num_heads=self.num_heads, head_dim=(self.model_ch)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.dec11 = BasicBlock(self.model_ch + self.model_ch, self.model_ch, num_heads=self.num_heads, head_dim=(self.model_ch)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)
        self.dec12 = BasicBlock(self.model_ch + self.model_ch, self.model_ch, num_heads=self.num_heads, head_dim=(self.model_ch)//num_heads, context_dim=self.context_dim, depth=self.depth, norm_num_groups=self.norm_num_groups)

    """TODO"""
    # method for make encoder layers
    def make_enc_layers(self, BasicBlock, num_basic_block, channel_mult):
        """
        method which makes ENCODER layer for Unet.
        :Basic Block: BasicBlock Class 
        """
        ch = self.model_ch

        layers = nn.ModuleList([])
        for level, mult_val in enumerate(channel_mult):
            # define Transformer head dimension
            head_dim = ch // self.num_heads
            # repeat basic block by num_basic_block
            if level < len(channel_mult)-1: # 마지막 level 이 아닐경우
                for _ in range(num_basic_block):
                    layers.append(BasicBlock(ch, ch, num_heads=self.num_heads, head_dim=head_dim, depth=self.depth, context_dim=self.context_dim,)) 
                    ch = mult_val * self.model_ch # update channel
                # DownSample
                layers.append(DownSample(ch, ch))

            else: # 마지막 level 일 경우
                for _ in range(num_basic_block):
                    layers.append(BasicBlock(ch, ch, use_transformer_block=False))

        return layers
    
    # method for make encoder layers
    def make_mid_layers(self, BasicBlock, channel_mult):
        """
        method which makes MID layer for Unet.
        :Basic Block: BasicBlock Class 
        """
        ch = channel_mult[-1] * self.model_ch
        head_dim = (ch * 2) // self.num_heads
        layers = [BasicBlock(ch, ch*2, num_heads=self.num_heads, head_dim=head_dim, depth=self.depth, context_dim=self.context_dim, mid_block=True, dropout=self.dropout)] 
        return layers
    
    #TODO 
    # skip connection 차원 어떻게 맞출 것인가?
    def make_dec_layers(self, num_basic_block, channel_mult):
        channel_mult = channel_mult[::-1]

        ch = channel_mult[0] * self.model_ch
        layers = nn.ModuleList([])
        for level, mult_val in enumerate(channel_mult):
            # define Transformer head dimension
            head_dim = ch // self.num_heads

            for i in range(num_basic_block + 1): # 두개의 Basic Block 과 DownSample 로부터 Skipconnection 을 받으므로 +1 필요
                layers.append(BasicBlock(ch, ch, num_heads=self.num_heads, head_dim=head_dim, depth=self.depth, context_dim=self.context_dim, dropout=self.dropout))
                ch = mult_val * self.model_ch

                if i == self.num_block:
                    layers.append(UpSample(ch))

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, condition_dict):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, 256) # self.context_dim
        

        label_emb = condition_dict["contents"]
        sty_emb = condition_dict["style"]
        context = torch.concat([label_emb, sty_emb], dim=1)
        context = self.cond_emb_linear(context)
        context = context.unsqueeze(dim=1)
        # sty_emb_t = self.sty_avgpool(condition_dict["style"])
        # b,c,_,_ = sty_emb_t.shape # b,c,h,w
        # sty_emb_t = sty_emb_t.view(b,c)
        # stroke_emb = self.stroke_linear(condition_dict["stroke"]).unsqueeze(dim=1)
        # context = torch.concat([stroke_emb_t, label_emb_t, sty_emb_t], dim=1)

        # context = torch.concat([label_emb, stroke_emb, sty_emb], dim=1)
        # context = torch.concat([label_emb, sty_emb], dim=1)

        """임시"""
        # enc level1
        x1 = self.input(x)
        x2 = self.enc1(x1, t, context)
        x3 = self.enc2(x2, t, context)
        # enc level2
        x4 = self.down1(x3)
        x5 = self.enc3(x4, t, context)
        x6 = self.enc4(x5, t, context)
        # enc level3
        x7 = self.down2(x6)
        x8 = self.enc5(x7, t, context)
        x9 = self.enc6(x8, t, context)
        # enc level4
        x10 = self.down3(x9)
        x11 = self.enc7(x10, t, context)
        x12 = self.enc8(x11, t, context)
        # mid
        out = self.mid(x12, t, context)
        # dec level4
        out = torch.cat([out, x12], dim=1)
        out = self.dec1(out, t, context)
        out = torch.cat([out, x11], dim=1)
        out = self.dec2(out, t, context)
        out = torch.cat([out, x10], dim=1)
        out = self.dec3(out, t, context)
        out = self.up1(out)
        # dec level3
        out = torch.cat([out, x9], dim=1)
        out = self.dec4(out, t, context)
        out = torch.cat([out, x8], dim=1)
        out = self.dec5(out, t, context)
        out = torch.cat([out, x7], dim=1)
        out = self.dec6(out, t, context)
        out = self.up2(out)
        # dec level2
        out = torch.cat([out, x6], dim=1)
        out = self.dec7(out, t, context)
        out = torch.cat([out, x5], dim=1)
        out = self.dec8(out, t, context)
        out = torch.cat([out, x4], dim=1)
        out = self.dec9(out, t, context)
        out = self.up3(out)
        # dec level1
        out = torch.cat([out, x3], dim=1)
        out = self.dec10(out, t, context)
        out = torch.cat([out, x2], dim=1)
        out = self.dec11(out, t, context)
        out = torch.cat([out, x1], dim=1)
        out = self.dec12(out, t, context)

        return self.out(out)


