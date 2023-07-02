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
from modules.model import UNet32,UNet128
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc
import wandb


# seed
seed = 7777

# graphic number
gpu_num = 0
image_size = 128
input_size = 64
batch_size = 4
num_classes = 11172
lr = 3e-4
n_epochs = 200
use_amp = True


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
    train_dirs = 'C:\Paper_Project\Hangul_Characters_Image64_GrayScale'

    # Set transform
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((input_size,input_size)),
        # torchvision.transforms.Grayscale(num_output_channels=1),
    # # #     torchvision.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(train_dirs,transform=transforms)

    #test set
    #n = range(0,len(dataset),5000)
    #dataset = Subset(dataset, n)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=8)

    #Set model
    model = UNet128(num_classes=num_classes).to(device)
    wandb.watch(model)

    #Set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    #Set loss function
    loss_func = nn.MSELoss()

    ### sty_encoder
    sty_encoder_path = 'C:\Paper_Project\weight\style_enc.pth'
    checkpoint = torch.load(sty_encoder_path, map_location='cpu')
    tmp_dict = {}
    for k, v in checkpoint.items():
        if k in model.sty_encoder.state_dict():
            tmp_dict[k] = v
    model.sty_encoder.load_state_dict(tmp_dict)

    # frozen sty_encoder
    for p in model.sty_encoder.parameters():
        p.requires_grad = False


    #Set diffusion
    diffusion = Diffusion(first_beta=1e-4,
                          end_beta=0.02,
                          noise_step=1000,
                          beta_schedule_type='cosine',
                          img_size=input_size,
                          device=device)


    for epoch_id in range(n_epochs):
        print(f"Epoch {epoch_id}/{n_epochs} Train..")
        
        pbar = tqdm(dataloader,desc=f"trian_{epoch_id}")
        tic = time()
        for i, (x, y) in enumerate(pbar):
            print('x1 : ', x.shape)
            x = x.to(device)
            y = y.to(device)
            t = diffusion.sample_t(x.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(x, t)
            if np.random.random() < 0.3:
                y = None
            print('x2 : ', x.shape)
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

        # Save

        # labels = torch.arange(num_classes).long().to(device)
        # sampled_images = diffusion.portion_sampling(model, n=len(labels),sampleImage_len = 36)
        # plot_images(sampled_images)
        # save_images(sampled_images, os.path.join(result_image_path, f"{epoch_id}.jpg"))
        # torch.save(model,os.path.join(result_model_path,f"model_{epoch_id}.pt"))
        # torch.save(model.state_dict(), os.path.join(result_model_path, f"ckpt_{epoch_id}.pt"))
        # torch.save(optimizer.state_dict(), os.path.join(result_model_path, f"optim_{epoch_id}.pt"))

    wandb.finish()
