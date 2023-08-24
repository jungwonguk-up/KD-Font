import torch, torchvision

import os
from matplotlib import pyplot as plt
from modules.diffusion import Diffusion
from modules.model import UNet128
from PIL import Image
import wandb

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    for idx,image in enumerate(images):
        grid = torchvision.utils.make_grid(image, **kwargs)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        im.save(f"{path}_{idx}.jpg")

# wandb init
wandb.init(project="diffusion_font_sampling", config={
    "learning_rate": 0.0003,
    "architecture": "UNET",
    "dataset": "HOJUN_KOREAN_FONT64",
    "notes":"content, stoke, style/ 64 x 64"
})

# sample path
sample_img_path = 'C:/Paper_Project/Hangul_Characters_Image64_GrayScale/가/62570_가.png'
sample_img = Image.open(sample_img_path)

# sampe to Tensor
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()],
                                       )
sample_img = trans(sample_img).to('cuda')
sample_img = torch.unsqueeze(sample_img,1)
sample_img = sample_img.repeat(18, 1,1,1)


print(sample_img.shape)


epoch_id = 69
result_image_path = os.path.join("results", 'font_noStrokeStyle_{}'.format(2))
num_classes = 11172
n = 36
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet128(num_classes=n, sample_img=sample_img).to(device)
ckpt = torch.load("C:/Paper_Project/Ko_diffusion/models/font_noStrokeStyle_2/ckpt_69.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(first_beta=1e-4,
                          end_beta=0.02,
                          noise_step=1000,
                          beta_schedule_type='cosine',
                          img_size=64, 
                          device=device)
# y = torch.Tensor([15] * n).long().to(device)
# x = diffusion.sampling(model, n, y, cfg_scale=3)

labels = torch.arange(num_classes).long().to(device)
sampled_images = diffusion.portion_sampling(model, n=len(labels),sampleImage_len = 36)
plot_images(sampled_images)
save_images(sampled_images, os.path.join(result_image_path, f"{epoch_id}"))