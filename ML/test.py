import os
import random
import wandb
import torch, torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from PIL import Image

from matplotlib import pyplot as plt
from modules.diffusion import Diffusion
from modules.utils import plot_images, test_save_images,make_stroke,stroke_to_char
from models.utils import UNet
from modules.utils import CharAttar
from modules.datasets import DiffusionDataset

batch_size = 8 #####
sampleImage_len = 25


num_classes = 11172 # 이게 문제인건가?
input_length = 100
contents_dim = 100
input_size = 64
mode = "new"
folder_name ="test_3"
train_dirs = 'sample_data'
sample_img_path = 'sample_img/d03fc0a9c3190dce.png'
style_path = "/root/paper_project/ML/weight/style_enc.pth"
csv_path = "/root/paper_project/Tools/MakeFont/diffusion_font_train.csv"

sampling_chars = "괴그기깅나는늘다도디러로를만버없에우워을자점하한했"

if __name__ == '__main__':
    wandb.init(project="diffusion_font_test_sampling", config={
                "learning_rate": 0.0003,
                "architecture": "UNET",
                "dataset": "HOJUN_KOREAN_FONT64",
                "notes":"content, yes_stoke, non_style/ 64 x 64, 420 dataset"
                },
            name = "self-attetnion condtion content stroke style_sampling 나눔손글씨강인한위로_갊") #####
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    ckpt = torch.load("weight/ckpt_290.pt")
    model.load_state_dict(ckpt)

    diffusion = Diffusion(first_beta=1e-4,
                              end_beta=0.02,
                              noise_step=1000,
                              beta_schedule_type='linear',
                              img_size=input_size,
                              device=device)
    
    # to do 데이터로더는 필요 없음 삭제! 
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    dataset = DiffusionDataset(csv_path=csv_path,transform=transforms)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=12)

    #sample_img
    sample_img = Image.open(sample_img_path)
    sample_img = transforms(sample_img).to(device)
    sample_img = torch.unsqueeze(sample_img,1)
    sample_img = sample_img.repeat(len(sampling_chars), 1, 1, 1)
    
    charAttar = CharAttar(num_classes=num_classes,device=device,style_path=style_path)
    
    sampled_images = diffusion.portion_sampling(model, sampling_chars, charAttar=charAttar, sample_img=sample_img)