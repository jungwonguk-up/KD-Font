import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from modules import UNet
import logging
# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(massage)s", level=logging.INFO, datefmt="%I%M%S")

device = "cuda" if torch.cuda.is_available() else "cpu"

class Diffusion():
    def __init__(self,
                 noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 img_size=256,
                 device=device):
        self.noise_step = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.beta_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        def beta_noise_schedule(self):
            return torch.linspace(self.beta_start, self.beta_end, self.noise_step)
        
        def noise_images(self, x, t):
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
            epsilon = torch.randn_like(x)

        def sample_timesteps(self, n):
            return torch.randint(low=1, high=self.noise_steps, size=(n,))
        
        def sample(self, model, n):
            logging.info(f"Sampling {n} new images...")

            model.eval()
            with torch.no_grad():
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
                for i in tqdm(range(self.noise_step, 0, -1), position=0):
                    t = (torch.ones(n) * i).long().to(self.device)
                    predicted_noise = model(x, t)
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    
                    x = 1 / torch.sqrt(alpha) * (x - (beta/torch.sqrt(1-alpha_hat))) * predicted_noise + beta * noise

                    model.train()
                    x = (x.clamp(-1, 1) + 1) / 2
                    x = (x * 255).type(torch.uint8)
                    return x


def train(args):
    setup_logging(args.run_name)

    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    # logger = SummaryWriter(os.path.join("runs", args.run_name))logger
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}")

        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps()


