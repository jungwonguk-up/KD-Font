import torch
import torch.nn.functional as F
import torch.nn as nn
from .style_encoder import style_enc_builder
from .stroke import StrokeEmbedding

C = 32
C_in = 1

from .attention import TrasformerBlock


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x_size = x.shape[-1]
        x = x.view(-1, self.channels, x_size * x_size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, x_size, x_size)
    

class Attention(nn.Module):
    """
    Basic Transformer Block include self attention, cross attention, feed forward network
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm1 = nn.LayerNorm([channels])
        self.norm2 = nn.LayerNorm([channels])
        self.norm3 = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, context):
        x_size = x.shape[-1]
        # c_size = context.shape[-1]
        x = x.view(-1, self.channels, x_size * x_size).swapaxes(1, 2)
        # context = context.view(-1, self.channels, c_size * c_size).swapaxes(1, 2)
        x_ln = self.norm1(x)
        attention_value1, _ = self.mha(x_ln, x_ln, x_ln) # self-attention
        attention_value1 = attention_value1 + x
        attention_value1 = self.norm2(attention_value1)
        attention_value2, _ = self.mha(attention_value1, context, context) # cross-attention
        attention_value2 = attention_value2 + attention_value1
        attention_value2 = self.norm3(attention_value2)
        attention_value2 = self.ff_self(attention_value2) + attention_value2
        return attention_value2.swapaxes(2, 1).view(-1, self.channels, x_size, x_size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    

class TransformerUnet128(nn.Module):
    def __init__(self, 
                 c_in=1, 
                 c_out=1, 
                 time_dim=256, 
                 context_dim=32,
                 num_classes=None, 
                 device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.attn1 = TrasformerBlock(in_channels=128, context_dim=context_dim)
        self.down2 = Down(128, 256)
        self.attn2 = TrasformerBlock(in_channels=256, context_dim=context_dim)
        self.down3 = Down(256, 256)
        self.attn3 = TrasformerBlock(in_channels=256, context_dim=context_dim)

        self.bot1 = DoubleConv(256, 512)
        self.attn4 = TrasformerBlock(in_channels=512, context_dim=context_dim)
        self.bot2 = DoubleConv(512, 512)
        self.attn5 = TrasformerBlock(in_channels=512, context_dim=context_dim)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.attn6 = TrasformerBlock(in_channels=128, context_dim=context_dim)
        self.up2 = Up(256, 64)
        self.attn7 = TrasformerBlock(in_channels=64, context_dim=context_dim)
        self.up3 = Up(128, 64)
        self.attn8 = TrasformerBlock(in_channels=64, context_dim=context_dim)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            # self.label_emb = nn.Embedding(num_classes, time_dim)
            self.sty_encoder = style_enc_builder(C_in, C).to(device)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y, context):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            # class 로 넣고 한줄로 처리 -> 인자 하나더 받아서 처리해라! 
            stroke_embedding = StrokeEmbedding('C:\Paper_Project\storke_txt.txt')
            stroke_embedding = stroke_embedding.embedding(y)
            label = y.unsqueeze(1)
            sty = self.sty_encoder(x)
            # Adjust the shapes of the tensors by repeating the necessary dimensions
            stroke_embedding = stroke_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 16, 16).cuda()  # Expand and repeat to match shape with sty tensor
            label = label.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 16, 16)  # Expand and repeat to match shape with sty tensor

            # Concatenate the tensors along the second dimension (channels)
            context = torch.cat((stroke_embedding, label, sty), dim=1)

            # print(context.shape)


        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.attn1(x2, context)
        x3 = self.down2(x2, t)
        x3 = self.attn2(x3, context)
        x4 = self.down3(x3, t)
        x4 = self.attn3(x4, context)

        x4 = self.bot1(x4)
        x4 = self.attn4(x4, context)
        x4 = self.bot2(x4)
        x4 = self.attn5(x4, context)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.attn6(x, context)
        x = self.up2(x, x2, t)
        x = self.attn7(x, context)
        x = self.up3(x, x1, t)
        x = self.attn8(x, context)
        output = self.outc(x)
        return output


class CrossAttnUNet128(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.attn1 = Attention(128)
        self.down2 = Down(128, 256)
        self.attn2 = Attention(256)
        self.down3 = Down(256, 256)
        self.attn3 = Attention(256)

        self.bot1 = DoubleConv(256, 512)
        self.attn4 = Attention(512)
        self.bot2 = DoubleConv(512, 512)
        self.attn5 = Attention(512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.attn6 = Attention(128)
        self.up2 = Up(256, 64)
        self.attn7 = Attention(64)
        self.up3 = Up(128, 64)
        self.attn8 = Attention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, context):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.attn1(x, context)
        x3 = self.down2(x2, t)
        x3 = self.attn2(x, context)
        x4 = self.down3(x3, t)
        x4 = self.attn3(x, context)

        x4 = self.bot1(x4)
        x4 = self.attn4(x, context)
        x4 = self.bot2(x4)
        x4 = self.attn5(x, context)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.attn6(x, context)
        x = self.up2(x, x2, t)
        x = self.attn7(x, context)
        x = self.up3(x, x1, t)
        x = self.attn8(x, context)
        output = self.outc(x)
        return output


class UNet128(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            # self.label_emb = nn.Embedding(num_classes, time_dim)
            self.sty_encoder = style_enc_builder(C_in, C).to(device)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            # class 로 넣고 한줄로 처리 -> 인자 하나더 받아서 처리해라! 
            stroke_embedding = StrokeEmbedding('C:\Paper_Project\storke_txt.txt')
            stroke_embedding = stroke_embedding.embedding(y)
            label = y.unsqueeze(1)
            sty = self.sty_encoder(x)
            # Adjust the shapes of the tensors by repeating the necessary dimensions
            stroke_embedding = stroke_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 16, 16).cuda()  # Expand and repeat to match shape with sty tensor
            label = label.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 16, 16)  # Expand and repeat to match shape with sty tensor

            # Concatenate the tensors along the second dimension (channels)
            context = torch.cat((stroke_embedding, label, sty), dim=1)

            # print(context.shape)


        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet32(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128)
        self.down3 = Down(128, 128)
        self.sa3 = SelfAttention(128)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = SelfAttention(64)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output