import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn as nn
import random

from models.utils import style_enc_builder, StyleEncoder

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


class CharAttar:
    def __init__(self,num_classes,device,mode,batch_size):
        self.num_classes = num_classes
        self.contents_dim = 100
        self.contents_emb = nn.Embedding(num_classes, self.contents_dim)
        self.device = device
        self.mode = mode
        self.batch_size = batch_size
    def make_stroke(self,contents):
        strokes_list = []
        for content in contents:
            content_code = ord(content)
            first_letter_code = 44032
            stroke = [0] * 68
            first_consonant_letter = int((content_code - first_letter_code) / 588)
            middle_consonant_letter = int(((content_code - first_letter_code) - (first_consonant_letter * 588)) / 28)
            last_consonant_letter = int((content_code - first_letter_code) - (first_consonant_letter * 588) - (middle_consonant_letter * 28))
            stroke[first_consonant_letter] = 1
            stroke[middle_consonant_letter + 19] = 1
            stroke[last_consonant_letter + 19 + 21] = 1
            strokes_list.append(stroke)
        return strokes_list

    # def set_charAttr_dim(mode):
    #     pass
    def make_charAttr(self,images,contents_index, contents):
        # contents_index = [int(content_index) for content_index in contents_index]
        style_encoder = style_enc_builder(3,3).to(self.device)
        contents_emb = None
        stroke =  None
        style_emb = None
        contents_p, stroke_p = random.random(), random.random()
        if self.mode == 1:

            if contents_p < 0.3:
                contents_emb = torch.zeros(self.batch_size,self.contents_dim)
            else:
                contents_emb = torch.FloatTensor(self.contents_emb(contents_index))

            if stroke_p < 0.3:
                stroke = torch.zeros(self.batch_size,68)
            else:
                stroke =  torch.FloatTensor(self.make_stroke(contents))

            style_emb = torch.zeros(self.batch_size,12288)

        elif self.mode == 2:
            if contents_p < 0.3:
                contents_emb = torch.zeros(self.batch_size,self.contents_dim)
            else:
                contents_emb = torch.FloatTensor(self.contents_emb(contents_index))

            if stroke_p < 0.3:
                stroke = torch.zeros(self.batch_size,68)
            else:
                stroke = torch.FloatTensor(self.make_stroke(contents))

            if contents_p < 0.3 and stroke_p < 0.3:
                style_emb = torch.zeros(self.batch_size,12288)
            else:
                style_emb = style_encoder(images)
                style_emb = torch.flatten(style_emb)

        return torch.cat([contents_emb,stroke,style_emb],dim=1)

