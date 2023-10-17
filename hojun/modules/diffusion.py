import wandb,math
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset,DataLoader
from .utils import CharAttar
class Diffusion:

    def __init__(self, first_beta, end_beta, beta_schedule_type, noise_step, img_size, device):
        self.first_beta = first_beta
        self.end_beta = end_beta
        self.beta_schedule_type = beta_schedule_type

        self.noise_step = noise_step

        self.beta_list = self.beta_schedule().to(device)

        self.alphas =  1. - self.beta_list
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0)


        self.img_size = img_size
        self.device = device

    def sample_t(self, batch_size):
        return torch.randint(1,self.noise_step,(batch_size,))

    def beta_schedule(self):
        if self.beta_schedule_type == "linear":
            return torch.linspace(self.first_beta, self.end_beta, self.noise_step)
        elif self.beta_schedule_type == "cosine":
            steps = self.noise_step + 1
            s = 0.008
            x = torch.linspace(0, self.noise_step, steps)
            alphas_cumprod = torch.cos(((x / self.noise_step) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif self.beta_schedule_type == "quadratic":
            return torch.linspace(self.first_beta ** 0.5, self.end_beta ** 0.5, self.noise_step) ** 2
        elif self.beta_schedule_type == "sigmoid":
            beta = torch.linspace(-6,-6,self.noise_step)
            return torch.sigmoid(beta) * (self.end_beta - self.first_beta) + self.first_beta


    def alpha_t(self, t):
        return self.alphas[t][:, None, None, None]

    def alpha_bar_t (self,t):
        return self.alpha_bars[t][:, None, None, None]

    def one_minus_alpha_bar(self,t):
        return (1. - self.alpha_bars[t])[:, None, None, None]

    def beta_t(self,t):
        return self.beta_list[t][:, None, None, None]

    def noise_images(self,x,t):
        epsilon = torch.randn_like(x)
        return torch.sqrt(self.alpha_bar_t(t)) * x + torch.sqrt(self.one_minus_alpha_bar(t)) * epsilon , epsilon

    def indexToChar(self,y):
        return chr(44032+y)
    def portion_sampling(self, model, n,sampleImage_len,dataset,mode,charAttar,sample_img, cfg_scale=3):
        example_images = []
        model.eval()
        with torch.no_grad():
            x_list = torch.randn((sampleImage_len, 1, self.img_size, self.img_size)).to(self.device)

            y_idx = list(range(n))[::math.floor(n/sampleImage_len)][:sampleImage_len]
            contents_index = torch.IntTensor(y_idx)
            contents = [dataset.dataset.classes[content_index] for content_index in contents_index]
            charAttr_list = charAttar.make_charAttr(sample_img, contents_index, contents,mode=3).to(self.device)

            pbar = tqdm(list(reversed(range(1, self.noise_step))),desc="sampling")
            for i in pbar:
                dataset = TensorDataset(x_list,charAttr_list)
                batch_size= 18
                dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
                predicted_noise = torch.tensor([]).to(self.device)
                uncond_predicted_noise = torch.tensor([]).to(self.device)
                for batch_x, batch_conditions in dataloader:
                    batch_t = (torch.ones(len(batch_x)) * i).long().to(self.device)
                    batch_noise = model(batch_x, batch_t, batch_conditions)
                    predicted_noise = torch.cat([predicted_noise,batch_noise],dim=0)
                    #uncodition
                    uncond_batch_noise = model(batch_x, batch_t, torch.zeros_like(batch_conditions))
                    uncond_predicted_noise = torch.cat([uncond_predicted_noise,uncond_batch_noise],dim = 0)

                if cfg_scale > 0:
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                t = (torch.ones(sampleImage_len) * i).long()
                a_t = self.alpha_t(t)
                aBar_t = self.alpha_bar_t(t)
                b_t = self.beta_t(t)

                if i > 1:
                    noise = torch.randn_like(x_list)
                else:
                    noise = torch.zeros_like(x_list)

                x_list = 1 / torch.sqrt(a_t) * (
                        x_list - ((1 - a_t) / (torch.sqrt(1 - aBar_t))) * predicted_noise) + torch.sqrt(
                    b_t) * noise
        for sample_image,sample_content in zip(x_list,contents):
            example_images.append(wandb.Image(sample_image, caption=f"Sample:{sample_content}"))
        wandb.log({
            "Examples": example_images
        })
        model.train()
        x_list = (x_list.clamp(-1, 1) + 1) / 2
        x_list = (x_list * 255).type(torch.uint8)
        return x_list

    def test_sampling(self, model,sampleImage_len,charAttr_list,cfg_scale=3):
        example_images = []
        model.eval()
        with torch.no_grad():
            x_list = torch.randn((sampleImage_len, 3, self.img_size, self.img_size)).to(self.device)
            pbar = tqdm(list(reversed(range(1, self.noise_step))),desc="sampling")
            for i in pbar:
                dataset = TensorDataset(x_list,charAttr_list)
                batch_size= 4
                dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
                predicted_noise = torch.tensor([]).to(self.device)
                uncond_predicted_noise = torch.tensor([]).to(self.device)
                for batch_x, batch_conditions in dataloader:
                    batch_t = (torch.ones(len(batch_x)) * i).long().to(self.device)
                    batch_noise = model(batch_x, batch_t, batch_conditions)
                    predicted_noise = torch.cat([predicted_noise,batch_noise],dim=0)
                    #uncodition
                    uncond_batch_noise = model(batch_x, batch_t, torch.zeros_like(batch_conditions))
                    uncond_predicted_noise = torch.cat([uncond_predicted_noise,uncond_batch_noise],dim = 0)

                if cfg_scale > 0:
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                t = (torch.ones(sampleImage_len) * i).long()
                a_t = self.alpha_t(t)
                aBar_t = self.alpha_bar_t(t)
                b_t = self.beta_t(t)

                if i > 1:
                    noise = torch.randn_like(x_list)
                else:
                    noise = torch.zeros_like(x_list)

                x_list = 1 / torch.sqrt(a_t) * (
                        x_list - ((1 - a_t) / (torch.sqrt(1 - aBar_t))) * predicted_noise) + torch.sqrt(
                    b_t) * noise
        model.train()
        x_list = (x_list.clamp(-1, 1) + 1) / 2
        x_list = (x_list * 255).type(torch.uint8)
        return x_list