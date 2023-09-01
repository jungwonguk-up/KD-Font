from datetime import datetime
from time import time
import numpy as np
import random, os
from glob import glob
import torch, torchvision
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

from torch.utils.data import random_split, Subset

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from modules.diffusion import Diffusion
from modules.model import UNet128,TransformerUnet128
from modules.condition import MakeCondition
from modules.style_encoder import style_enc_builder
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import gc
import wandb
import os

# seed
seed = 7777

# graphic number
gpu_num = 1
image_size = 64
input_size = 64
batch_size = 16
num_classes = 11172
lr = 3e-4
n_epochs = 200
use_amp = True
resume_train = False
file_num = 13
stroke_text_path = "/home/hojun/Documents/code/Kofont2/KoFont-Diffusion/storke_txt.txt"
style_enc_path = "/home/hojun/Documents/code/Kofont2/KoFont-Diffusion/weight/style_enc.pth"
start_epoch = 0

# os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def save_images(images, path, **kwargs):
    for idx,image in enumerate(images):
        grid = torchvision.utils.make_grid(image, **kwargs)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        im.save(path)

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


if __name__ == '__main__':
    #Set save file
    result_image_path = os.path.join("results", 'font_noStrokeStyle_{}'.format(file_num))
    result_model_path = os.path.join("models", 'font_noStrokeStyle_{}'.format(file_num))
    os.makedirs(result_image_path, exist_ok=True)
    os.makedirs(result_model_path, exist_ok=True)
 
    # wandb init
    wandb.init(project="diffusion_font_32_test", config={
        "learning_rate": 0.0003,
        "architecture": "UNET",
        "dataset": "HOJUN_KOREAN_FONT64",
        "notes":"content, non_stoke, non_style/ 32 x 32"
    })


    # Set random seed, deterministic
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:28"
    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set data directory
    train_dirs = 'C:\Paper_Project\Hangul_Characters_Image64_radomSampling420_GrayScale'

    # Set transform
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.Grayscale(num_output_channels=1),
    # # #     torchvision.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(train_dirs,transform=transforms)

    #sample_img
    sample_img_path = 'C:/Paper_Project/Hangul_Characters_Image64_radomSampling420_GrayScale/갊/62570_갊.png'
    sample_img = Image.open(sample_img_path)
    sample_img = transforms(sample_img).to(device)
    sample_img = torch.unsqueeze(sample_img,1)
    sample_img = sample_img.repeat(18, 1, 1, 1)

    #test set
    n = range(0,len(dataset),50)
    print("len : ",n)
    dataset = Subset(dataset, n)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    
    #sample_img
    sample_img_path = '/home/hojun/PycharmProjects/diffusion_font/code/KoFont-Diffusion/hojun/make_font/data/Hangul_Characters_Image64_radomSampling420_GrayScale/갊/62570_갊.png'
    sample_img = Image.open(sample_img_path)
    sample_img = transforms(sample_img).to(device)
    sample_img = torch.unsqueeze(sample_img,1)
    sample_img = sample_img.repeat(18, 1, 1, 1)

    if resume_train:
        #Set model
        model = TransformerUnet128(num_classes=num_classes,device = device).to(device)
        # model = UNet128(num_classes=num_classes).to(device)
        wandb.watch(model)

        #Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        #load weight
        start_epoch = 9
        model.load_state_dict(torch.load(f'/home/hojun/Documents/code/Kofont2/KoFont-Diffusion/Ko_diffusion/models/font_noStrokeStyle_12/ckpt_2_{start_epoch}.pt'))

        #load optimzer
        optimizer.load_state_dict(torch.load(f'/home/hojun/Documents/code/Kofont2/KoFont-Diffusion/Ko_diffusion/models/font_noStrokeStyle_12/ckpt_2_{start_epoch}.pt'))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model = model.to(device)

    else:
        #Set model
        model = TransformerUnet128(num_classes=num_classes, context_dim=256,device = device).to(device) # 여기는 왜 256이지?
        # model = UNet128(num_classes=num_classes).to(device)
        wandb.watch(model)

        #Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    #Set loss function
    loss_func = nn.MSELoss()

    ### stroke
    make_condition = MakeCondition(num_classes=num_classes,
                                    stroke_text_path=stroke_text_path,
                                    style_enc_path=style_enc_path,
                                    data_classes=dataset.dataset.classes,
                                    language='korean',
                                    device=device
                                )
    
    #Set diffusion
    diffusion = Diffusion(first_beta=1e-4,
                          end_beta=0.02,
                          noise_step=1000,
                          beta_schedule_type='cosine', # stable diffusion 확인
                          img_size=input_size,
                          device=device)
    
    for epoch_id in range(start_epoch,n_epochs):
        print(f"Epoch {epoch_id}/{n_epochs} Train..")
        
        pbar = tqdm(dataloader,desc=f"trian_{epoch_id}")
        tic = time()
        for i, (image, content) in enumerate(pbar):
            # print('x1 : ', x.shape)
            image = image.to(device)
            condition = make_condition.make_condition(images = image, indexs = content, mode=4).to(device) 
            
            t = diffusion.sample_t(image.shape[0]).to(device)
            image_t, noise = diffusion.noise_images(image, t)
            predicted_noise = model(x = image_t, condition = condition, t= t) # 원래 이미지 -> 스타일 인코더
            loss = loss_func(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc = time()
        wandb.log({"train_mse_loss": loss,'train_time':toc-tic})
        pbar.set_postfix(MSE=loss.item())


        if epoch_id % 10 == 0 :

            # Save

            labels = torch.arange(num_classes).long().to(device)
            sampled_images = diffusion.portion_sampling(model, n=len(labels), sampleImage_len = 36, sty_img = sample_img, make_condition = make_condition) # self, model, n,sampleImage_len, cfg_scale=0, sty_img = None, make_condition = None
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join(result_image_path, f"{epoch_id}.jpg"))
            torch.save(model,os.path.join(result_model_path,f"model_nonstyle_{epoch_id}.pt"))
            torch.save(model.state_dict(), os.path.join(result_model_path, f"ckpt_nonstyle_{epoch_id}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(result_model_path, f"optim_nonstyle_{epoch_id}.pt"))

    wandb.finish()
