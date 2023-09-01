import torch, torchvision

import os
from matplotlib import pyplot as plt
from modules.diffusion import Diffusion
from modules.model import UNet128, TransformerUnet128
from modules.condition import MakeCondition
from PIL import Image
import wandb

gpu_num = 0
file_num = 12
epoch_id = 1
num_classes = 11172
n = 36
stroke_text_path = "C:\Paper_Project\storke_txt.txt"
style_enc_path = "C:\Paper_Project\weight\style_enc.pth"

# os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == '__main__':
    
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

    result_image_path = os.path.join("results", 'font_noStrokeStyle_{}'.format(file_num))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    train_dirs = 'C:\Paper_Project\Hangul_Characters_Image64_radomSampling420_GrayScale'
    dataset = torchvision.datasets.ImageFolder(train_dirs)
    
    make_condition = MakeCondition(num_classes=num_classes,
                                    stroke_text_path=stroke_text_path,
                                    style_enc_path=style_enc_path,
                                    data_classes=dataset.classes,
                                    language='korean',
                                    device=device
                                )
    
    # sample path
    sample_img_path = 'C:/Paper_Project/Hangul_Characters_Image64_radomSampling420_GrayScale/갊/62570_갊.png'
    sample_img = Image.open(sample_img_path)

    # sampe to Tensor
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])
    sample_img = trans(sample_img).to(device)
    sample_img = torch.unsqueeze(sample_img,1)
    sample_img = sample_img.repeat(18, 1, 1, 1)

    model = TransformerUnet128(num_classes=n, context_dim=256,device=device).to(device)
    ckpt = torch.load('C:/Paper_Project/Ko_diffusion/models/font_noStrokeStyle_12/ckpt_nonstyle_50.pt')
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
    sampled_images = diffusion.portion_sampling(model, n=len(dataset.classes),sampleImage_len = 36, sty_img = sample_img, make_condition= make_condition)
    plot_images(sampled_images)
    save_images(sampled_images, os.path.join(result_image_path, f"{epoch_id}"))