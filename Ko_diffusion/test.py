import torch, torchvision

import os
from matplotlib import pyplot as plt
from modules.diffusion import Diffusion
from modules.model import UNet128, TransformerUnet128
from PIL import Image
import wandb

gpu_num = 1
file_num = 5

# os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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





epoch_id = 69
result_image_path = os.path.join("results", 'font_noStrokeStyle_{}'.format(file_num))
num_classes = 11172
n = 36
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sample path
sample_img_path = '/home/hojun/PycharmProjects/diffusion_font/code/KoFont-Diffusion/hojun/make_font/data/Hangul_Characters_Image64_radomSampling420_GrayScale/갊/62570_갊.png'
sample_img = Image.open(sample_img_path)

# sampe to Tensor
trans = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
sample_img = trans(sample_img).to(device)
sample_img = torch.unsqueeze(sample_img,1)
sample_img = sample_img.repeat(18, 1,1,1)

model = TransformerUnet128(num_classes=n, context_dim=256,device=device).to(device)
ckpt = torch.load(f"/home/hojun/Documents/code/KoFont-Diffusion/Ko_diffusion/models/font_noStrokeStyle_{file_num}/ckpt_2_30.pt")
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
sampled_images = diffusion.portion_sampling(model, n=len(labels),sampleImage_len = 36, sty_img = sample_img)
plot_images(sampled_images)
save_images(sampled_images, os.path.join(result_image_path, f"{epoch_id}"))