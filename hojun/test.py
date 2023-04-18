import torch

import os
from matplotlib import pyplot as plt
from modules.diffusion import Diffusion
from models.utils import UNet

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

n = 36
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(num_classes=n).to(device)
ckpt = torch.load("./results/models/font_noStrokeStyle_Unet64_image64_2/ckpt_20.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
y = torch.Tensor([15] * n).long().to(device)
x = diffusion.sampling(model, n, y, cfg_scale=3)
plot_images(x)