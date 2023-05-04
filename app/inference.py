from io import BytesIO
import torch
import random
from PIL import Image
# from typing import Optional

import os
from matplotlib import pyplot as plt
from diffusion_modules.diffusion import Diffusion
from diffusion_modules.utils import plot_images, test_save_images, make_stroke, stroke_to_char
from diffusion_models.utils import UNet


class DiffusionModel:
    def __init__(self,
                 num_classes: int = 420,
                 contents_dim: int = 100,
                 input_size: int = 64,
                 device: str = "cuda"):

        self.num_classes = num_classes
        self.contents_dim = contents_dim
        self.input_size = input_size
        self.device = device

        self.model = UNet().to(self.device)
        self.diffusion = Diffusion(first_beta=1e-4,
                                   end_beta=0.02,
                                   noise_step=1000,
                                   beta_schedule_type='linear',
                                   img_size=self.input_size,
                                   device=self.device)
        
    def load_state_dict(self, path: str):
        """
        지정된 경로에서 weight 를 불러와 model 에 로드

        Args:
            path: str = 저장된 checkpoint.pt 의 경로 
        """
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)

    def random_sampling(self, input_length: int = 1) -> Image.Image:
        """
        input_length 만큼 임의로 문자 이미지를 생성
        return : Image.Image 객체

        Args:
            input_length: int = 랜덤하게 생성할 문자의 수
        """
        contents_emb = torch.zeros(input_length, self.contents_dim)

        first= [random.randint(0,18) for _ in range(input_length)]
        middle = [random.randint(19,39) for _ in range(input_length)]
        last = [random.randint(40,67) for _ in range(input_length)]

        strokes = torch.Tensor([[0 for _ in range(68)] for _ in range(input_length)])

        for idx in range(input_length):
            strokes[idx][first[idx]], strokes[idx][middle[idx]], strokes[idx][last[idx]] = 1, 1, 1
        char_list = stroke_to_char(strokes)

        style_emb = torch.zeros(input_length, 12288)

        y = torch.cat([contents_emb, strokes, style_emb], dim=1).to(self.device)
        x = self.diffusion.test_sampling(self.model, input_length, y, cfg_scale=3)

        img = Image.fromarray(x[0].permute(1, 2, 0).cpu().numpy())

        return img


    def manual_sampling(self, char: str = None) -> Image.Image:
        """
        char 문자 이미지 생성
        return : Image.Image 객체

        Args:
            char: str = 생성하고 싶은 단어 또는 문자 (길이 = 1)
        """

        if char == None:
            raise ValueError("message: 글자를 넣으세요")
        
        assert len(char) <= 1 # 임의로 제한
    
        input_length = len(char)
        char_list = []
        for i in char:
            char_list.append(i)

        contents_emb = torch.zeros(input_length, self.contents_dim)
        strokes = torch.Tensor(make_stroke(char_list))
        style_emb = torch.zeros(input_length, 12288)
        y = torch.cat([contents_emb, strokes, style_emb], dim=1).to(self.device)
        x = self.diffusion.test_sampling(self.model, len(strokes), y, cfg_scale=3)

        # TODO : 여러개의 이미지 반환
        img = Image.fromarray(x[0].permute(1, 2, 0).cpu().numpy())

        return img


diffusion_model = DiffusionModel(device="cuda")


if __name__ == "__main__":
    diffusion_model.load_state_dict(path="C:/Users/gih54/Desktop/diffusion/ckpt_290.pt")
    diffusion_model.manual_sampling(char='캺')
