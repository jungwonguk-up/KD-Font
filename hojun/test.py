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
batch_size = 8 #####
sampleImage_len = 36


num_classes = 420
input_length = 100
contents_dim = 100
input_size = 64
mode = "new"
folder_name ="test_3"

train_dirs = '/home/hojun/PycharmProjects/diffusion_font/code/KoFont-Diffusion/hojun/make_font/data/Hangul_Characters_Image64_radomSampling420_GrayScale'
sample_img_path = f'{train_dirs}/갊/나눔손글씨강인한위로_갊.png'

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
    ckpt = torch.load("/home/hojun/Documents/code/Kofont5/KoFont-Diffusion2/hojun/results/models/font_noStrokeStyle_Unet64_image420_3/ckpt_290.pt")
    model.load_state_dict(ckpt)

    diffusion = Diffusion(first_beta=1e-4,
                              end_beta=0.02,
                              noise_step=1000,
                              beta_schedule_type='linear',
                              img_size=input_size,
                              device=device)
    
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(train_dirs,transform=transforms)

    # test set
    n = range(0,len(dataset),10)
    dataset = Subset(dataset, n)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=12)

    #sample_img
    sample_img = Image.open(sample_img_path)
    sample_img = transforms(sample_img).to(device)
    sample_img = torch.unsqueeze(sample_img,1)
    sample_img = sample_img.repeat(sampleImage_len, 1, 1, 1)
    
    if mode == "random":
        contents_emb = torch.zeros(input_length,contents_dim)

        first= [random.randint(0,18) for _ in range(input_length)]
        middle = [random.randint(19,39) for _ in range(input_length)]
        last = [random.randint(40,67) for _ in range(input_length)]

        strokes = torch.Tensor([[0 for _ in range(68)] for _ in range(input_length)])

        for idx in range(input_length):
            strokes[idx][first[idx]], strokes[idx][middle[idx]], strokes[idx][last[idx]] = 1, 1, 1
        char_list = stroke_to_char(strokes)

        style_emb = torch.zeros(input_length,12288)

        y = torch.cat([contents_emb, strokes, style_emb], dim=1).to(device)
        x = diffusion.test_sampling(model, input_length, y, cfg_scale=3)

    elif mode == "manual":
        char_list = ['가,나,다,라,마,바,사,아,자,차,카,타,파,하']
        contents_emb = torch.zeros(input_length, contents_dim)
        strokes = make_stroke(char_list)
        style_emb = torch.zeros(input_length, 12288)
        y = torch.cat([contents_emb, strokes, style_emb], dim=1).to(device)
        x = diffusion.test_sampling(model,len(strokes), y, cfg_scale=3)
        
    elif mode == "new":
        charAttar = CharAttar(num_classes=num_classes,device=device)
        sampled_images = diffusion.portion_sampling(model, n=len(dataset.dataset.classes),sampleImage_len = sampleImage_len,dataset=dataset,mode =mode,charAttar=charAttar,sample_img=sample_img)