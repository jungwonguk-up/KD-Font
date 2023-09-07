import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from PIL import Image
import random, os
from collections import OrderedDict

from models.style_encoder import style_enc_builder

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
        self.device = device
        self.contents_dim = 100
        self.contents_emb = nn.Embedding(num_classes, self.contents_dim)
        self.style_enc = self.make_style_enc("/home/hojun/Documents/code/Kofont5/KoFont-Diffusion2/hojun/style_enc.pth")
    
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
    
    def make_style_enc(self,style_enc_path):
        C ,C_in = 32, 1
        sty_encoder = style_enc_builder(C_in, C)
        checkpoint = torch.load(style_enc_path, map_location=self.device)
        tmp_dict = {}
        for k, v in checkpoint.items():
            if k in sty_encoder.state_dict():
                tmp_dict[k] = v
        sty_encoder.load_state_dict(tmp_dict)
        # frozen sty_encoder
        for p in sty_encoder.parameters():
            p.requires_grad = False
        return sty_encoder.to(self.device)

    # def set_charAttr_dim(mode):
    #     pass
    def make_charAttr(self,images,contents_index, contents,mode):
        input_length = images.shape[0]
        # contents_index = [int(content_index) for content_index in contents_index]
        # style_encoder = style_enc_builder(1,32).to(self.device)

        contents_emb = None
        stroke =  None
        style = None
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

            style = torch.zeros(input_length,128)

        elif mode == 2:
            
            if contents_p < 0.3:
                contents_emb = torch.zeros(input_length,self.contents_dim)
            else:
                contents_emb = torch.FloatTensor(self.contents_emb(contents_index))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length,68)
            else:
                stroke = torch.FloatTensor(self.make_stroke(contents))

            if contents_p < 0.3 and stroke_p < 0.3:
                style = torch.zeros(input_length,128)
            else:
                style = self.style_enc(images)
                style = F.adaptive_avg_pool2d(style, (1, 1))
                style = style.view(input_length, -1).cpu()


        elif mode == 3: #test
            contents_emb = torch.FloatTensor(self.contents_emb(contents_index))
            stroke = torch.FloatTensor(self.make_stroke(contents))
            style = self.style_enc(images)
            style = F.adaptive_avg_pool2d(style, (1, 1))
            style = style.view(input_length, -1).cpu()
            
        elif mode == 4:
            contents_emb = torch.FloatTensor(self.contents_emb(contents_index))
            stroke = torch.FloatTensor(self.make_stroke(contents))
            style = torch.zeros(input_length,128)
            
        return torch.cat([contents_emb,stroke,style],dim=1)

