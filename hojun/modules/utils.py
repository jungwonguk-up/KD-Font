import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn as nn
import random, os
from collections import OrderedDict

from models.utils import style_enc_builder, StyleEncoder

def save_images(images, path, **kwargs):
    for idx,image in enumerate(images):
        grid = torchvision.utils.make_grid(image, **kwargs)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        im.save(path)

def test_save_images(images, char_list,folder_name):
    to_path = os.path.join("./results/test/",folder_name)
    os.makedirs(to_path,exist_ok=True)

    for image,char in zip(images,char_list):
        im = Image.fromarray(image.permute(1,2,0).cpu().numpy())
        im.save(os.path.join(to_path,char+".png"),"PNG")

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def stroke_to_char(strokes):
    char_list = []
    for storke in strokes:
        storke_index = []
        for idx in range(len(storke)):
            if storke[idx] == 1:
                storke_index.append(idx)
        char_list.append((storke_index[0]*588)+((storke_index[1]-19)*28)+(storke_index[2]-40)+44032)
    char_list = [chr(x) for x in char_list]
    return char_list

def make_stroke(contents):
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

class CharAttar:
    def __init__(self,num_classes,device):
        self.num_classes = num_classes
        self.contents_dim = 100
        self.contents_emb = nn.Embedding(num_classes, self.contents_dim)
        self.device = device
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
    def make_charAttr(self,images,contents_index, contents,mode):
        input_length = images.shape[0]
        # contents_index = [int(content_index) for content_index in contents_index]
        # style_encoder = style_enc_builder(1,32).to(self.device)

        contents_emb = None
        stroke =  None
        style_emb = None
        contents_p, stroke_p = random.random(), random.random()
        if mode == 1:

            if contents_p < 0.3:
                contents_emb = torch.zeros(input_length,self.contents_dim)
            else:
                contents_emb = torch.FloatTensor(self.contents_emb(contents_index))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length,68)
            else:
                stroke =  torch.FloatTensor(self.make_stroke(contents))

            style_emb = torch.zeros(input_length,12288)

        elif mode == 2:
            style_encoder = style_enc_builder(3, 3).to(self.device)
            style_econder_dict = torch.load(
                '/home/hojun/PycharmProjects/diffusion_font/code/KoFont-Diffusion/hojun/results/models/style_weight/style_encoder_weight.pth',
                map_location=torch.device(self.device))
            style_econder_dict = style_econder_dict
            print(style_econder_dict)
            change_style_econder_dict = OrderedDict()
            for key, weight in style_econder_dict.items():
                name = ".".join(key.split('.')[1:])
                change_style_econder_dict[name] = weight
            #     if "layers.2.gc.k_proj.weight" in key:
            # # print(name,key)
            style_encoder.load_state_dict(change_style_econder_dict, strict=False)
            # style_encoder.load_state_dict(style_econder_dict)

            if contents_p < 0.3:
                contents_emb = torch.zeros(input_length,self.contents_dim)
            else:
                contents_emb = torch.FloatTensor(self.contents_emb(contents_index))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length,68)
            else:
                stroke = torch.FloatTensor(self.make_stroke(contents))

            if contents_p < 0.3 and stroke_p < 0.3:
                style_emb = torch.zeros(input_length,12288)
            else:
                style_emb = style_encoder(images)
                style_emb = torch.flatten(style_emb)

        elif mode == 3: #test
            contents_emb = torch.FloatTensor(self.contents_emb(contents_index))
            stroke = torch.FloatTensor(self.make_stroke(contents))
            style_emb = torch.zeros(input_length, 12288)
        return torch.cat([contents_emb,stroke,style_emb],dim=1)

