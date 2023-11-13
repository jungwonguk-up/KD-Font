import os
from tqdm import tqdm
import math
import random
import pandas as pd
import numpy as np
from PIL import Image

import torch, torchvision
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset,Dataset
# from model import UNet, Diffusion, CharAttar
from functools import partial


class DiffusionDataset(Dataset):
    def __init__(self, csv_path, transform =None):
        self.transform = transform
        self.csv_data = pd.read_csv(os.path.join(csv_path,"diffusion_font_train.csv"))
        self.x_file_name = self.csv_data.iloc[:,0]
        self.x_path = self.csv_data.iloc[:,1]
        self.y = self.csv_data.iloc[:,2]
        self.labels = np.unique(self.y)
        self.y_to_label = self.make_y_to_label()
        self.label_to_y = self.make_label_to_y()
        self.y_labels = self.make_y_labels()

    
    def make_y_to_label(self):
        y_to_label_dict = {}
        for label, value in enumerate(self.labels):
            y_to_label_dict[value] = label
        return y_to_label_dict
    
    def make_label_to_y(self):
        label_to_y_dict = {}
        for label, value in enumerate(self.labels):
            label_to_y_dict[label] = value
        return label_to_y_dict
    
    def make_y_labels(self):
        y_labels = []
        for y_ch in self.y:
            y_labels.append(self.y_to_label[y_ch])
        return y_labels
    
    def __len__(self):
        return len(self.x_path)
    
    def __getitem__(self, id_: int):
        filename = self.x_file_name[id_]
        x = Image.open(self.x_path[id_])
        transform_x = self.transform(x)
        y_ch = self.y[id_]
        
        return transform_x, y_ch, filename
            
            

class Diffusion:    
    def __init__(self, first_beta, end_beta, beta_schedule_type, noise_step, img_size, device):
        self.first_beta = first_beta
        self.end_beta = end_beta
        self.beta_schedule_type = beta_schedule_type

        self.noise_step = noise_step

        self.beta_list = self.beta_schedule().to(device)

        self.alphas =  1. - self.beta_list
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)


        self.img_size = img_size
        self.device = device

    def sample_t(self, batch_size):
        return torch.randint(1,self.noise_step,(batch_size,))

    def beta_schedule(self):
        if self.beta_schedule_type == "linear":
            return torch.linspace(self.first_beta, self.end_beta, self.noise_step)
        elif self.beta_schedule_type == "cosine":
            steps = self.noise_step + 1
            s = 0.008
            x = torch.linspace(0, self.noise_step, steps)
            alphas_cumprod = torch.cos(((x / self.noise_step) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif self.beta_schedule_type == "quadratic":
            return torch.linspace(self.first_beta ** 0.5, self.end_beta ** 0.5, self.noise_step) ** 2
        elif self.beta_schedule_type == "sigmoid":
            beta = torch.linspace(-6,-6,self.noise_step)
            return torch.sigmoid(beta) * (self.end_beta - self.first_beta) + self.first_beta


    def alpha_t(self, t):
        return self.alphas[t][:, None, None, None]

    def alpha_bar_t (self,t):
        return self.alpha_bars[t][:, None, None, None]

    def one_minus_alpha_bar(self,t):
        return (1. - self.alpha_bars[t])[:, None, None, None]

    def beta_t(self,t):
        return self.beta_list[t][:, None, None, None]

    def noise_images(self,x,t):
        epsilon = torch.randn_like(x)
        return torch.sqrt(self.alpha_bar_t(t)) * x + torch.sqrt(self.one_minus_alpha_bar(t)) * epsilon , epsilon

    def indexToChar(self,y):
        return chr(44032+y)
    def portion_sampling(self, model, n,sampleImage_len,dataset,mode,charAttar,sample_img, cfg_scale=3):
        example_images = []
        model.eval()
        with torch.no_grad():
            x_list = torch.randn((sampleImage_len, 1, self.img_size, self.img_size)).to(self.device)

            y_idx = list(range(n))[::math.floor(n/sampleImage_len)][:sampleImage_len]
            contents_index = torch.IntTensor(y_idx)
            contents = [dataset.label_to_y[int(content_index)] for content_index in contents_index]
            charAttr_list = charAttar.make_charAttr(sample_img, contents_index, contents,mode=3).to(self.device)

            pbar = tqdm(list(reversed(range(1, self.noise_step))),desc="sampling")
            for i in pbar:
                dataset = TensorDataset(x_list,charAttr_list)
                batch_size= 18
                dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
                predicted_noise = torch.tensor([]).to(self.device)
                uncond_predicted_noise = torch.tensor([]).to(self.device)
                for batch_x, batch_conditions in dataloader:
                    batch_t = (torch.ones(len(batch_x)) * i).long().to(self.device)
                    batch_noise = model(batch_x, batch_t, batch_conditions)
                    predicted_noise = torch.cat([predicted_noise,batch_noise],dim=0)
                    #uncodition
                    uncond_batch_noise = model(batch_x, batch_t, torch.zeros_like(batch_conditions))
                    uncond_predicted_noise = torch.cat([uncond_predicted_noise,uncond_batch_noise],dim = 0)

                if cfg_scale > 0:
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                t = (torch.ones(sampleImage_len) * i).long()
                a_t = self.alpha_t(t)
                aBar_t = self.alpha_bar_t(t)
                b_t = self.beta_t(t)

                if i > 1:
                    noise = torch.randn_like(x_list)
                else:
                    noise = torch.zeros_like(x_list)

                x_list = 1 / torch.sqrt(a_t) * (
                        x_list - ((1 - a_t) / (torch.sqrt(1 - aBar_t))) * predicted_noise) + torch.sqrt(
                    b_t) * noise
        model.train()
        x_list = (x_list.clamp(-1, 1) + 1) / 2
        x_list = (x_list * 255).type(torch.uint8)
        return x_list

    def test_sampling(self, model,sampleImage_len,charAttr_list,cfg_scale=3):
        example_images = []
        model.eval()
        with torch.no_grad():
            x_list = torch.randn((sampleImage_len, 1, self.img_size, self.img_size)).to(self.device)
            pbar = tqdm(list(reversed(range(1, self.noise_step))),desc="sampling")
            for i in pbar:
                dataset = TensorDataset(x_list,charAttr_list)
                batch_size= 4
                dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
                predicted_noise = torch.tensor([]).to(self.device)
                uncond_predicted_noise = torch.tensor([]).to(self.device)
                for batch_x, batch_conditions in dataloader:
                    batch_t = (torch.ones(len(batch_x)) * i).long().to(self.device)
                    batch_noise = model(batch_x, batch_t, batch_conditions)
                    predicted_noise = torch.cat([predicted_noise,batch_noise],dim=0)
                    #uncodition
                    uncond_batch_noise = model(batch_x, batch_t, torch.zeros_like(batch_conditions))
                    uncond_predicted_noise = torch.cat([uncond_predicted_noise,uncond_batch_noise],dim = 0)

                if cfg_scale > 0:
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                t = (torch.ones(sampleImage_len) * i).long()
                a_t = self.alpha_t(t)
                aBar_t = self.alpha_bar_t(t)
                b_t = self.beta_t(t)

                if i > 1:
                    noise = torch.randn_like(x_list)
                else:
                    noise = torch.zeros_like(x_list)

                x_list = 1 / torch.sqrt(a_t) * (
                        x_list - ((1 - a_t) / (torch.sqrt(1 - aBar_t))) * predicted_noise) + torch.sqrt(
                    b_t) * noise
        model.train()
        x_list = (x_list.clamp(-1, 1) + 1) / 2
        x_list = (x_list * 255).type(torch.uint8)
        return x_list

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
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

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
    def __init__(self, in_channels, out_channels, time_dim=256, charAttr_dim=12456):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.time_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                out_channels
            ),
        )

        self.condition_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                charAttr_dim,
                out_channels
            ),
        )

    def forward(self, x, t,charAttr):
        x = self.maxpool_conv(x)
        emb = self.time_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        charAttr_emb = self.condition_layer(charAttr)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb + charAttr_emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, charAttr_dim=12456):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.time_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                out_channels
            ),
        )
        self.condition_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                charAttr_dim,
                out_channels
            ),
        )


    def forward(self, x, skip_x, t, charAttr):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        time_emb = self.time_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        charAttr_emb = self.condition_layer(charAttr)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + time_emb + charAttr_emb

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, charAttr_dim = 296, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.charAttr_dim = charAttr_dim

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, time_dim=self.time_dim, charAttr_dim=self.charAttr_dim)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256,time_dim=self.time_dim, charAttr_dim=self.charAttr_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, time_dim=self.time_dim, charAttr_dim=self.charAttr_dim)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, time_dim=self.time_dim, charAttr_dim=self.charAttr_dim)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64, time_dim=self.time_dim, charAttr_dim=self.charAttr_dim)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64, time_dim=self.time_dim, charAttr_dim=self.charAttr_dim)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, time, charAttr):
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_dim)

        # if y is not None:
        #     time += self.contents_emb(y)


        x1 = self.inc(x)
        x2 = self.down1(x1, time, charAttr)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, time, charAttr)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, time, charAttr)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, time, charAttr)
        x = self.sa4(x)
        x = self.up2(x, x2, time, charAttr)
        x = self.sa5(x)
        x = self.up3(x, x1, time, charAttr)
        x = self.sa6(x)
        output = self.outc(x)
        return output

class GlobalContext(nn.Module):
    """ Global-context """
    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        C_bottleneck = int(C * bottleneck_ratio)
        w_norm = w_norm_dispatch(w_norm)
        self.k_proj = w_norm(nn.Conv2d(C, 1, 1))
        self.transform = nn.Sequential(
            w_norm(nn.Linear(C, C_bottleneck)),
            nn.LayerNorm(C_bottleneck),
            nn.ReLU(),
            w_norm(nn.Linear(C_bottleneck, C))
        )

    def forward(self, x):
        # x: [B, C, H, W]
        context_logits = self.k_proj(x)  # [B, 1, H, W]
        context_weights = F.softmax(context_logits.flatten(1), dim=1)  # [B, HW]
        context = torch.einsum('bci,bi->bc', x.flatten(2), context_weights)
        out = self.transform(context)

        return out[..., None, None]

class GCBlock(nn.Module):
    """ Global-context block """
    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        self.gc = GlobalContext(C, bottleneck_ratio, w_norm)

    def forward(self, x):
        gc = self.gc(x)
        return x + gc

class TLU(nn.Module):
    """ Thresholded Linear Unit """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return torch.max(x, self.tau)

    def extra_repr(self):
        return 'num_features={}'.format(self.num_features)


# NOTE generalized version
class FilterResponseNorm(nn.Module):
    """ Filter Response Normalization """
    def __init__(self, num_features, ndim, eps=None, learnable_eps=False):
        """
        Args:
            num_features
            ndim
            eps: if None is given, use the paper value as default.
                from paper, fixed_eps=1e-6 and learnable_eps_init=1e-4.
            learnable_eps: turn eps to learnable parameter, which is recommended on
                fully-connected or 1x1 activation map.
        """
        super().__init__()
        if eps is None:
            if learnable_eps:
                eps = 1e-4
            else:
                eps = 1e-6

        self.num_features = num_features
        self.init_eps = eps
        self.learnable_eps = learnable_eps
        self.ndim = ndim

        self.mean_dims = list(range(2, 2+ndim))

        self.weight = nn.Parameter(torch.ones([1, num_features] + [1]*ndim))
        self.bias = nn.Parameter(torch.zeros([1, num_features] + [1]*ndim))
        if learnable_eps:
            self.eps = nn.Parameter(torch.as_tensor(eps))
        else:
            self.register_buffer('eps', torch.as_tensor(eps))

    def forward(self, x):
        # normalize
        nu2 = x.pow(2).mean(self.mean_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # modulation
        x = x * self.weight + self.bias

        return x

    def extra_repr(self):
        return 'num_features={}, init_eps={}, ndim={}'.format(
                self.num_features, self.init_eps, self.ndim)


FilterResponseNorm1d = partial(FilterResponseNorm, ndim=1, learnable_eps=True)
FilterResponseNorm2d = partial(FilterResponseNorm, ndim=2)

def split_dim(x, dim, n_chunks):
    shape = x.shape
    assert shape[dim] % n_chunks == 0
    return x.view(*shape[:dim], n_chunks, shape[dim] // n_chunks, *shape[dim+1:])


def weights_init(init_type='default'):
    """ Adopted from FUNIT """
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=2**0.5)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=2**0.5)
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    return init_fun


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    return nn.utils.spectral_norm(module)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


def dispatcher(dispatch_fn):
    def decorated(key, *args):
        if callable(key):
            return key

        if key is None:
            key = 'none'

        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def norm_dispatch(norm):
    return {
        'none': nn.Identity,
        'in': partial(nn.InstanceNorm2d, affine=False),  # false as default
        'bn': nn.BatchNorm2d,
        'frn': FilterResponseNorm2d
    }[norm.lower()]


@dispatcher
def w_norm_dispatch(w_norm):
    # NOTE Unlike other dispatcher, w_norm is function, not class.
    return {
        'spectral': spectral_norm,
        'none': lambda x: x
    }[w_norm.lower()]


@dispatcher
def activ_dispatch(activ, norm=None):
    if norm_dispatch(norm) == FilterResponseNorm2d:
        # use TLU for FRN
        activ = 'tlu'

    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
        "tlu": TLU
    }[activ.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {
        "zero": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d
    }[pad_type.lower()]


class ParamBlock(nn.Module):
    def __init__(self, C_out, shape):
        super().__init__()
        w = torch.randn((C_out, *shape))
        b = torch.randn((C_out,))
        self.shape = shape
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        b = self.b.reshape((1, *self.b.shape, 1, 1, 1)).repeat(x.size(0), 1, *self.shape)
        return self.w*x + b


class LinearBlock(nn.Module):
    """ pre-active linear block """
    def __init__(self, C_in, C_out, norm='none', activ='relu', bias=True, w_norm='none',
                dropout=0.):
        super().__init__()
        activ = activ_dispatch(activ, norm)
        if norm.lower() == 'bn':
            norm = nn.BatchNorm1d
        elif norm.lower() == 'frn':
            norm = FilterResponseNorm1d
        elif norm.lower() == 'none':
            norm = nn.Identity
        else:
            raise ValueError(f"LinearBlock supports BN only (but {norm} is given)")
        w_norm = w_norm_dispatch(w_norm)
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        self.linear = w_norm(nn.Linear(C_in, C_out, bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)


class ConvBlock(nn.Module):
    """ pre-active conv block """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none',
                activ='relu', bias=True, upsample=False, downsample=False, w_norm='none',
                pad_type='zero', dropout=0., size=None):
        # 1x1 conv assertion
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample

        assert ((norm == FilterResponseNorm2d) == (activ == TLU)), "Use FRN and TLU together"

        if norm == FilterResponseNorm2d and size == 1:
            self.norm = norm(C_in, learnable_eps=True)
        else:
            self.norm = norm(C_in)
        if activ == TLU:
            self.activ = activ(C_in)
        else:
            self.activ = activ()
        if dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    """ Pre-activate ResBlock with spectral normalization """
    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False,
                norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0.,
                scale_var=False):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ,
                            upsample=upsample, w_norm=w_norm, pad_type=pad_type,
                            dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ,
                            w_norm=w_norm, pad_type=pad_type, dropout=dropout)

        # XXX upsample / downsample needs skip conv?
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        """
        normal: pre-activ + convs + skip-con
        upsample: pre-activ + upsample + convs + skip-con
        downsample: pre-activ + convs + downsample + skip-con
        => pre-activ + (upsample) + convs + (downsample) + skip-con
        """
        out = x

        out = self.conv1(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        # skip-con
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)
        return out


class Upsample1x1(nn.Module):
    """Upsample 1x1 to 2x2 using Linear"""
    def __init__(self, C_in, C_out, norm='none', activ='relu', w_norm='none'):
        assert norm.lower() != 'in', 'Do not use instance norm for 1x1 spatial size'
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.proj = ConvBlock(
            C_in, C_out*4, 1, 1, 0, norm=norm, activ=activ, w_norm=w_norm
        )

    def forward(self, x):
        # x: [B, C_in, 1, 1]
        x = self.proj(x)  # [B, C_out*4, 1, 1]
        B, C = x.shape[:2]
        return x.view(B, C//4, 2, 2)


class HourGlass(nn.Module):
    """U-net like hourglass module"""
    def __init__(self, C_in, C_max, size, n_downs, n_mids=1, norm='none', activ='relu',
                w_norm='none', pad_type='zero'):
        """
        Args:
            C_max: maximum C_out of left downsampling block's output
        """
        super().__init__()
        assert size == n_downs ** 2, "HGBlock assume that the spatial size is downsampled to 1x1."
        self.C_in = C_in

        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, w_norm=w_norm, pad_type=pad_type)

        self.lefts = nn.ModuleList()
        c_in = C_in
        for i in range(n_downs):
            c_out = min(c_in*2, C_max)
            self.lefts.append(ConvBlk(c_in, c_out, downsample=True))
            c_in = c_out

        # 1x1 conv for mids
        self.mids = nn.Sequential(
            *[
                ConvBlk(c_in, c_out, kernel_size=1, padding=0)
                for _ in range(n_mids)
            ]
        )

        self.rights = nn.ModuleList()
        for i, lb in enumerate(self.lefts[::-1]):
            c_out = lb.C_in
            c_in = lb.C_out
            channel_in = c_in*2 if i else c_in  # for channel concat
            if i == 0:
                block = Upsample1x1(channel_in, c_out, norm=norm, activ=activ, w_norm=w_norm)
            else:
                block = ConvBlk(channel_in, c_out, upsample=True)
            self.rights.append(block)

    def forward(self, x):
        features = []
        for lb in self.lefts:
            x = lb(x)
            features.append(x)

        assert x.shape[-2:] == torch.Size((1, 1))

        for i, (rb, lf) in enumerate(zip(self.rights, features[::-1])):
            if i:
                x = torch.cat([x, lf], dim=1)
            x = rb(x)

        return x

class StyleEncoder(nn.Module):
    def __init__(self, layers, out_shape):
        super().__init__()

        self.layers = nn.Sequential(*layers)
        self.out_shape = out_shape

    def forward(self, x):
        style_feat = self.layers(x)
        
        return style_feat


def style_enc_builder(C_in, C, norm='none', activ='relu', pad_type='reflect', skip_scale_var=False):
    
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'),
        ConvBlk(C*1, C*2, 3, 1, 1, downsample=True),
        GCBlock(C*2),
        ConvBlk(C*2, C*4, 3, 1, 1, downsample=True),
        CBAM(C*4)
    ]

    out_shape = (C*4, 32, 32)

    return StyleEncoder(layers, out_shape)


class CharAttar:
    def __init__(self,num_classes,device,style_path):
        self.num_classes = num_classes
        self.device = device
        self.contents_dim = 100
        self.contents_emb = nn.Embedding(num_classes, self.contents_dim)
        self.style_enc = self.make_style_enc(os.path.join(style_path,"style_enc.pth"))
        self.style_conv = nn.Sequential(
                            nn.Conv2d(128,128,16),
                            nn.SiLU(),
                        ).to(device)
    
    def make_stroke(self,contents):
        strokes_list = []
        for content in contents:
            content_code = ord(content)
            first_letter_code = 44032
            stroke = [0] * 68
            first_consonant_letter = int((content_code - first_letter_code) / 588)
            middle_consonant_letter = int(((content_code - first_letter_code) - (first_consonant_letter * 588)) / 28)
            last_consonant_letter = int((content_code - first_letter_code) - (first_consonant_letter * 588) - (middle_consonant_letter * 28))
            stroke[first_consonant_letter] = 1
            stroke[middle_consonant_letter + 19] = 1
            stroke[last_consonant_letter + 19 + 21] = 1
            strokes_list.append(stroke)
        return strokes_list
    
    def make_style_enc(self,style_enc_path):
        C ,C_in = 32, 1
        sty_encoder = style_enc_builder(C_in, C)
        checkpoint = torch.load(style_enc_path, map_location=self.device)
        tmp_dict = {}
        for k, v in checkpoint.items():
            if k in sty_encoder.state_dict():
                tmp_dict[k] = v
        sty_encoder.load_state_dict(tmp_dict)
        # frozen sty_encoder
        for p in sty_encoder.parameters():
            p.requires_grad = False
        return sty_encoder.to(self.device)
    
    def make_ch_to_index(self,contents):
        index_list = []
        first_letter_code = 44032
        for content in contents:
            content_code = ord(content)
            index_list.append(content_code - first_letter_code)
        index_list = torch.IntTensor(index_list)
        return index_list
            
    # def set_charAttr_dim(mode):
    #     pass
    def make_charAttr(self,images, contents,mode):
        input_length = images.shape[0]
        # contents_index = [int(content_index) for content_index in contents_index]
        # style_encoder = style_enc_builder(1,32).to(self.device)

        contents_emb = None
        stroke =  None
        style = None
        contents_p, stroke_p = random.random(), random.random()
        if mode == 1:

            if contents_p < 0.3:
                contents_emb = torch.zeros(input_length,self.contents_dim)
            else:
                contents_emb = torch.FloatTensor(self.contents_emb(self.make_ch_to_index(contents)))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length,68)
            else:
                stroke =  torch.FloatTensor(self.make_stroke(contents))

            style = torch.zeros(input_length,128)

        elif mode == 2:
            
            if contents_p < 0.3:
                contents_emb = torch.zeros(input_length,self.contents_dim)
            else:
                contents_emb = torch.FloatTensor(self.contents_emb(self.make_ch_to_index(contents)))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length,68)
            else:
                stroke = torch.FloatTensor(self.make_stroke(contents))

            if contents_p < 0.3 and stroke_p < 0.3:
                style = torch.zeros(input_length,128)
            else:
                style = self.style_enc(images)
                style = F.adaptive_avg_pool2d(style, (1, 1))
                # style = self.style_conv(style)
                style = style.view(input_length, -1).cpu()
                


        elif mode == 3: #test
            contents_emb = torch.FloatTensor(self.contents_emb(self.make_ch_to_index(contents)))
            stroke = torch.FloatTensor(self.make_stroke(contents))
            style = self.style_enc(images)
            style = F.adaptive_avg_pool2d(style, (1, 1))
            # style = self.style_conv(style)
            style = style.view(input_length, -1).cpu()
            
        elif mode == 4:
            contents_emb = torch.FloatTensor(self.contents_emb(self.make_ch_to_index(contents)))
            stroke = torch.FloatTensor(self.make_stroke(contents))
            style = torch.zeros(input_length,128)
            
        return torch.cat([contents_emb,stroke,style],dim=1)