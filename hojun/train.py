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
from modules.utils import save_images, plot_images,CharAttar

from models.utils import UNet

import gc

import wandb
# # seed
# seed = 7777

# graphic number #####
gpu_num = 1 #####
input_size = 64
batch_size = 8 #####
num_classes = 36 #####
lr = 3e-4 #####
n_epochs = 300 #####
start_epoch = 10
mode = 1 # mode_1 : with contents, stroke, mode_2 : with contents, stroke, style #####

use_amp = True
resume_train = False




if __name__ == '__main__':
    #Set save file
    file_number= "Unet64_image420_2"
    result_image_path = os.path.join("results", "images", 'font_noStrokeStyle_{}'.format(file_number))
    result_model_path = os.path.join("results", "models", 'font_noStrokeStyle_{}'.format(file_number))
    if os.path.exists(result_model_path):
        print("file_exist")
        exit()
    os.makedirs(result_image_path, exist_ok=True)
    os.makedirs(result_model_path, exist_ok=True)

    # wandb init
    # wandb.init(project="diffusion_font_32_test", config={
    #     "learning_rate": 0.0003,
    #     "architecture": "UNET",
    #     "dataset": "HOJUN_KOREAN_FONT64",
    #     "notes":"content, yes_stoke, non_style/ 64 x 64, 420 dataset"
    # }) #####
    wandb.init(mode="disabled")


    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set data directory
    train_dirs = './make_font/data/Hangul_Characters_Image64_36' #####

    # Set transform
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size,input_size)),
    # #     torchvision.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(train_dirs,transform=transforms)

    # test set
    # n = range(0,len(dataset),100)
    # dataset = Subset(dataset, n)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



    if resume_train:
        #Set model
        model = UNet()

        #Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        #load weight
        model.load_state_dict(torch.load('./models/font_noStrokeStyle_Unet64_image64_2/ckpt_90.pt'))

        #load optimzer
        optimizer.load_state_dict(torch.load('./models/font_noStrokeStyle_Unet64_image64_2/optim_90.pt'))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model = model.to(device)
    else:
        # Set model
        model = UNet().to(device)
        # Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    wandb.watch(model)

    #Set loss function
    loss_func = nn.MSELoss()

    #Set diffusion
    diffusion = Diffusion(first_beta=1e-4,
                          end_beta=0.02,
                          noise_step=1000,
                          beta_schedule_type='linear',
                          img_size=input_size,
                          device=device)


    for epoch_id in range(n_epochs):
        if resume_train:
            epoch_id += start_epoch

        print(f"Epoch {epoch_id}/{n_epochs} Train..")

        pbar = tqdm(dataloader,desc=f"trian_{epoch_id}")
        tic = time()
        for i, (images, contents_index) in enumerate(pbar):
            images = images.to(device)

            contents = [dataset.classes[content_index] for content_index in contents_index]
            charAttar = CharAttar(num_classes=num_classes,device=device)

            charAttr_list = charAttar.make_charAttr(images, contents_index,contents,mode=2).to(device)

            t = diffusion.sample_t(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, t, charAttr_list)
            loss = loss_func(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc = time()
        wandb.log({"train_mse_loss": loss,'train_time':toc-tic})

        pbar.set_postfix(MSE=loss.item())


        if epoch_id % 10 == 0 :
            labels = torch.arange(num_classes).long().to(device)
            sampled_images = diffusion.portion_sampling(model, n=len(labels),sampleImage_len = 36,dataset=dataset,mode =mode,charAttar=charAttar)
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join(result_image_path, f"{epoch_id}.jpg"))
            torch.save(model,os.path.join(result_model_path,f"model_{epoch_id}.pt"))
            torch.save(model.state_dict(), os.path.join(result_model_path, f"ckpt_{epoch_id}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(result_model_path, f"optim_{epoch_id}.pt"))

    wandb.finish()
