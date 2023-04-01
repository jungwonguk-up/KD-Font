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
from modules.model import UNet32,UNet64
import gc

import wandb
# # seed
# seed = 7777

# graphic number
gpu_num = 1
image_size = 1024
input_size = 32
batch_size = 128
num_classes = 11172
lr = 3e-4
n_epochs = 200

use_amp = True
resume_train = False


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

# def memorydel(listobj):
#     try:
#         for obj in listobj:
#             del obj
#     except Exception as e:
#         print(e)
#     try:
#         del listobj
#     except Exception as e:
#         print(e)
#     gc.collect()
# def memorydel_all(listobj):
#     for x in listobj:
#         try:
#             memorydel(x)
#         except Exception as e:
#             print(e)
#     gc.collect()
#     torch.cuda.empty_cache()

if __name__ == '__main__':
    #Set save file
    result_image_path = os.path.join("results", 'font_noStrokeStyle_{}'.format(2))
    result_model_path = os.path.join("models", 'font_noStrokeStyle_{}'.format(2))
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
    # torch.cuda.manual_seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set data directory
    train_dirs = '/home/hojun/PycharmProjects/diffusion_font/code/make_font/Hangul_Characters_Image64'

    # Set transform
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size,input_size)),
    # #     torchvision.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(train_dirs,transform=transforms)

    #test set
    n = range(0,len(dataset),5000)
    dataset = Subset(dataset, n)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if resume_train:
        #Set model
        model = UNet64(num_classes=num_classes)

        #Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        #load weight
        model.load_state_dict(torch.load('/home/hojun/PycharmProjects/diffusion_font/code/diffusion/code/models/font_noStrokeStyle_2/ckpt_2.pt'))

        #load optimzer
        optimizer.load_state_dict(torch.load('/home/hojun/PycharmProjects/diffusion_font/code/diffusion/code/models/font_noStrokeStyle_2/optim_2.pt'))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model = model.to(device)
    else:
        # Set model
        model = UNet64(num_classes=num_classes).to(device)
        # Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    wandb.watch(model)

    #Set loss function
    loss_func = nn.MSELoss()

    #Set diffusion
    diffusion = Diffusion(first_beta=1e-4,
                          end_beta=0.02,
                          noise_step=1000,
                          beta_schedule_type='cosine',
                          img_size=input_size,
                          device=device)


    for epoch_id in range(n_epochs):
        if resume_train:
            epoch_id += 3

        print(f"Epoch {epoch_id}/{n_epochs} Train..")

        pbar = tqdm(dataloader,desc=f"trian_{epoch_id}")
        tic = time()
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            t = diffusion.sample_t(x.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(x, t)
            if np.random.random() < 0.3:
                y = None
            predicted_noise = model(x_t, t, y)
            loss = loss_func(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc = time()
        wandb.log({"train_mse_loss": loss,'train_time':toc-tic})
        # memorydel_all(x)
        # # memorydel_all(y)
        # # memorydel_all(t)
        # memorydel_all(x_t)
        # memorydel_all(predicted_noise)
        pbar.set_postfix(MSE=loss.item())


        #if epoch_id % 10 == 0 :
        labels = torch.arange(num_classes).long().to(device)
        sampled_images = diffusion.portion_sampling(model, n=len(labels),sampleImage_len = 36)
        # plot_images(sampled_images)
        save_images(sampled_images, os.path.join(result_image_path, f"{epoch_id}.jpg"))
        torch.save(model,os.path.join(result_model_path,f"model_{epoch_id}.pt"))
        torch.save(model.state_dict(), os.path.join(result_model_path, f"ckpt_{epoch_id}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(result_model_path, f"optim_{epoch_id}.pt"))

    wandb.finish()
