import random, os,sys
from time import time
from tqdm import tqdm
from PIL import Image
from glob import glob

import numpy as np
import torch, torchvision
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

from modules.diffusion import Diffusion
from modules.utils import CharAttar, save_images, load_yaml
from modules.datasets import DiffusionDataset, DiffusionSamplingDataset

from models.utils import UNet

import gc

import wandb


# # seed
# seed = 7777

# graphic number #####
# config['gpu_num'] = 0 #####
# config['input_size'] = 64
# config['batch_size'] = 8 #####
# config['num_classes'] = 11172 #####
# config['lr'] = 3e-4 #####
# config['n_epochs'] = 300 #####
# config['start_epoch'] = 10
# config['mode'] = 2 # mode_1 : with contents, stroke, mode_2 : with contents, stroke, style #####
# config['sampling_chars'] = "괴그기깅나는늘다도디러로를만버없에우워을자점하한했"
# config['noise_step'] = 1000
# config['cfg_scale'] = 3


# config['resume_train'] = False



if __name__ == '__main__':
    # Load config
    config_path = os.path.join(prj_dir, 'config', 'train.yaml')
    config = load_yaml(config_path)
    
    # Set path
    train_dirs = os.path.join(prj_dir,config['train_dirs'])
    sample_img_path  = os.path.join(prj_dir,config['sample_img_path'])
    csv_path = os.path.join(prj_dir,config['csv_path'])
    style_path = os.path.join(prj_dir,config['style_path'])
    
    result_image_path = os.path.join(prj_dir,config['result_path'], "images", config['folder_name'])
    result_model_path = os.path.join(prj_dir,config['result_path'], "models", config['folder_name'])
    
    if os.path.exists(result_model_path):
        print("file_exist")
        exit()
    os.makedirs(result_image_path, exist_ok=True)
    os.makedirs(result_model_path, exist_ok=True)

    if config['wandb']:
        wandb.init(project="diffusion_font_test", config={
                    "learning_rate": 0.0003,
                    "architecture": "UNET",
                    "dataset": "HOJUN_KOREAN_FONT64",
                    "notes":"content, yes_stoke, non_style/ 64 x 64, 420 dataset"
                    },
                   name = "self-attetnion condtion content stroke GAP 25")
    else:
        wandb.init(mode="disabled")


    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set data directory
    # train_dirs = '/home/hojun/PycharmProjects/diffusion_font/code/KoFont-Diffusion/hojun/make_font/data/Hangul_Characters_Image64_radomSampling420_GrayScale' #####

    # Set transform
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    
    train_dataset = DiffusionDataset(csv_path=csv_path,transform=transforms)
    sampling_dataset = DiffusionSamplingDataset(sampling_img_path=sample_img_path,sampling_chars=config['sampling_chars'],img_size=config['input_size'],device=device,transforms=transforms )
    
    # test set
    # n = range(0,len(dataset),10)
    # dataset = Subset(dataset, n)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=12)
    # test_dataloader = DataLoader(sampling_dataset,batch_size=config['batch_size'],shuffle=False)
    
    sample_img = Image.open(sample_img_path)
    sample_img = transforms(sample_img).to(device)
    sample_img = torch.unsqueeze(sample_img,1)
    sample_img = sample_img.repeat(len(config['sampling_chars']), 1, 1, 1)

    
    if config['resume_train']:
        #Set model
        model = UNet()

        #Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

        #load weight
        model.load_state_dict(torch.load(config['model_path']))

        #load optimzer
        optimizer.load_state_dict(torch.load(config['optim_path']))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model = model.to(device)
    else:
        # Set model
        model = UNet().to(device)
        # Set optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    wandb.watch(model)

    #Set loss function
    loss_func = nn.MSELoss()

    #Set diffusion
    diffusion = Diffusion(first_beta=config['first_beta'],
                          end_beta=config['end_beta'],
                          noise_step=config['noise_step'],
                          beta_schedule_type='linear',
                          img_size=config['input_size'],
                          device=device)


    charAttar = CharAttar(num_classes=config['num_classes'],device=device,style_path=style_path)
    
    for epoch_id in range(config['n_epochs']):
        if config['resume_train']:
            epoch_id += config['start_epoch']

        print(f"Epoch {epoch_id}/{config['n_epochs']} Train..")

        train_pbar = tqdm(train_dataloader,desc=f"trian_{epoch_id}")
        tic = time()
        for i, (train_images, train_contents_ch, train_filename) in enumerate(train_pbar):
            train_images = train_images.to(device)

            train_charAttr_list = charAttar.make_charAttr(train_images, train_contents_ch, mode=config['mode']).to(device)

            t = diffusion.sample_t(train_images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(train_images, t)

            train_predicted_noise = model(x_t, t, train_charAttr_list)
            loss = loss_func(noise, train_predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc = time()

        train_pbar.set_postfix(MSE=loss.item())
        
        if epoch_id % 10 == 0 :
            sampled_images = diffusion.portion_sampling(model, config['sampling_chars'],charAttar=charAttar,sample_img=sample_img,batch_size=config['batch_size'])
            model.train()
            
            # plot_images(sampled_images)
            # save_images(sample_all_x, os.path.join(result_image_path, f"{epoch_id}.jpg"))
            torch.save(model,os.path.join(result_model_path,f"model_{epoch_id}.pt"))
            torch.save(model.state_dict(), os.path.join(result_model_path, f"ckpt_{epoch_id}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(result_model_path, f"optim_{epoch_id}.pt"))
        else:
            wandb.log({"train_mse_loss": loss,'train_time':toc-tic})

    wandb.finish()
