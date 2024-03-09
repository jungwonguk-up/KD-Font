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
from torch.optim.swa_utils import AveragedModel

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from modules.diffusion import Diffusion
from modules.model import UNet128,TransformerUnet128

from modules.res_unet_model import Unet

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
gpu_num = 0
image_size = 64
input_size = 64
batch_size = 12
num_classes = 11172
lr = 1e-4
n_epochs = 802
use_amp = True
resume_train = False
file_num = "style_enc_2_ema"
train_dirs = "H:/data/Hangul_Characters_Image64_radomSampling420_GrayScale"
sample_img_path_1 = f'{train_dirs}/갊/62570_갊.png'
sample_img_path_2 = f'{train_dirs}/갊/나눔손글씨김유이체_갊.png'
stroke_text_path = "./text_weight/storke_txt.txt"
style_enc_path = "./text_weight/korean_styenc.ckpt"

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
    # # wnadb disable mode select
    os.environ["WANDB_DISABLED"] = "False"

    #Set save file
    result_image_path = os.path.join("results", 'font_noStrokeStyle_{}'.format(file_num))
    result_model_path = os.path.join("models", 'font_noStrokeStyle_{}'.format(file_num))
    os.makedirs(result_image_path, exist_ok=True)
    os.makedirs(result_model_path, exist_ok=True)
    
    ## wandb init
    wandb.init(project="cross_attention_font_test",
            #    name="Label Only (Linear) + t + (Cross Attention) lr 8e-5 ~400epoch",
               name="style_enc_2_ema",
               config={"learning_rate": 0.0001,
                       "architecture": "UNET",
                       "dataset": "HOJUN_KOREAN_FONT64",
                       "notes":"content, non_stoke, non_style/ 32 x 32"})
    # wandb.init(mode = "disabled")
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


    # train_dirs = "H:/data/hangle_image64_gray_small"

    # Set transform
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.Grayscale(num_output_channels=1),
    # # #     torchvision.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(train_dirs,transform=transforms)

    #test set
    n = range(0,len(dataset),10)
    print("len : ",n)
    dataset = Subset(dataset, n)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    #sample_img
    sample_img_1, sample_img_2 = Image.open(sample_img_path_1), Image.open(sample_img_path_2)
    sample_img_1, sample_img_2 = transforms(sample_img_1).to(device), transforms(sample_img_2).to(device)
    sample_img_1, sample_img_2 = torch.unsqueeze(sample_img_1, 1), torch.unsqueeze(sample_img_2, 1)
    sample_img_1, sample_img_2 = sample_img_1.repeat(8, 1, 1, 1), sample_img_2.repeat(8, 1, 1, 1)


    #Set model
    model = Unet(model_channels=128, context_dim=128, device=device).to(device)
    #Set EMA model
    ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))


    #Set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if resume_train:
        #load weight
        # start_epoch = 370
        model.load_state_dict(torch.load(f'./models/font_noStrokeStyle_{file_num}/ckpt_2_{start_epoch}.pt'))
        ema_model.load_state_dict(torch.load(f'./models/font_noStrokeStyle_{file_num}/ema_ckpt_2_{start_epoch}.pt.pt'))

        #load optimzer
        optimizer.load_state_dict(torch.load(f'./models/font_noStrokeStyle_{file_num}/optim_2_{start_epoch}.pt'))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model = model.to(device)


    wandb.watch(model)

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
    
    cond_mode = 3
    
    for epoch_id in range(start_epoch,n_epochs):
        print(f"Epoch {epoch_id}/{n_epochs} Train..")

        if cond_mode == 3 and epoch_id > 400:
            cond_mode = 1
        
        pbar = tqdm(dataloader, desc=f"trian_{epoch_id}", ncols=120)
        tic = time()
        for i, (image, content) in enumerate(pbar):
            # print('x1 : ', x.shape)
            image = image.to(device)
            # condition = make_condition.make_condition(images = image,indexs = content,mode=1).to(device)
            condition_dict = make_condition.make_condition(images=image, indexs=content, mode=cond_mode)
            
            t = diffusion.sample_t(image.shape[0]).to(device)
            image_t, noise = diffusion.noise_images(image, t)
            # predicted_noise = model(x = image_t, condition = condition, t= t) # 원래 이미지 -> 스타일 인코더
            predicted_noise = model(x=image_t, t=t, condition_dict = condition_dict)
            loss = loss_func(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model)

        toc = time()
        wandb.log({"train_mse_loss": loss,'train_time':toc-tic}, step=epoch_id, commit=True)
        pbar.set_postfix(MSE=loss.item())

        if epoch_id % 10 == 0 :
        # Save
            labels = torch.arange(num_classes).long().to(device)
            sampled_images1 = diffusion.portion_sampling(ema_model, n=len(dataset.dataset.classes), sampleImage_len=8, sty_img_1=sample_img_1, sty_img_2=sample_img_2, make_condition=make_condition, step=epoch_id, log_name="EMA")
            sampled_images2 = diffusion.portion_sampling(model, n=len(dataset.dataset.classes), sampleImage_len=8, sty_img_1=sample_img_1, sty_img_2=sample_img_2, make_condition=make_condition, step=epoch_id, log_name="None-EMA")
            # plot_images(sampled_images)
            save_images(sampled_images1, os.path.join(result_image_path, f"{epoch_id}.jpg"))
            save_images(sampled_images2, os.path.join(result_image_path, f"{epoch_id}.jpg"))
            torch.save(model,os.path.join(result_model_path,f"model_2_{epoch_id}.pt"))
            torch.save(model.state_dict(), os.path.join(result_model_path, f"ckpt_2_{epoch_id}.pt"))
            torch.save(ema_model.state_dict(), os.path.join(result_model_path, f"ema_ckpt_2_{epoch_id}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(result_model_path, f"optim_2_{epoch_id}.pt"))

    # Update bn statistics for the ema_model at the end ???
    torch.optim.swa_utils.update_bn(dataloader, ema_model)

    wandb.finish()
