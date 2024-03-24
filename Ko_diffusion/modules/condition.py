import random
import torch
import torch.nn as nn
from modules.style_encoder import style_enc_builder, StyleEncoder2

class Korean_StrokeEmbedding:
    def __init__(self,txt_path,classes):
        self.emd_stroke_list = self.read_stroke_txt(txt_path)
        self.classes = classes

    def read_stroke_txt(self, txt_path):
        emd_stroke_list = []
        read_txt = open(txt_path, 'r')
        for emd in read_txt:
            emd_stroke_list.append(emd[:-1])
        return emd_stroke_list

    def embedding(self, indexs):
        stroke_embedding = []
        for index in indexs:
            tmp = []
            unicode_diff = ord(self.classes[index]) - 44032# ord('가')
            for check in self.emd_stroke_list[unicode_diff]:
                tmp.append(int(check))
            stroke_embedding.append(tmp)
        return stroke_embedding

class MakeCondition:
    # 모든 label 이름 바꾸기
    def __init__(self, num_classes, stroke_text_path, style_enc_path, data_classes, language, device):
        self.device = device
        self.dataset_classes = data_classes
        self.num_classes = num_classes
        self.contents_dim = 128 
        self.contents_emb = nn.Embedding(num_classes, self.contents_dim)
        self.korean_stroke_emb = Korean_StrokeEmbedding(txt_path=stroke_text_path,classes=self.dataset_classes)
        # self.style_enc = self.make_style_enc(style_enc_path)
        self.style_enc = StyleEncoder2()
        self.language = language

        self.load_style_weight(style_enc_path)

    # def make_style_enc(self,style_enc_path):
    #     C ,C_in = 32, 1
    #     sty_encoder = style_enc_builder(C_in, C)
    #     checkpoint = torch.load(style_enc_path, map_location=self.device)
    #     tmp_dict = {}
    #     for k, v in checkpoint.items():
    #         if k in sty_encoder.state_dict():
    #             tmp_dict[k] = v
    #     sty_encoder.load_state_dict(tmp_dict)
    #     # frozen sty_encoder
    #     for p in sty_encoder.parameters():
    #         p.requires_grad = False
    #     return sty_encoder.to(self.device)
        
    def load_style_weight(self, style_enc_path):
        checkpoint = torch.load(style_enc_path, map_location=self.device)
        tmp_dict = {}
        for k, v in checkpoint.items():
            if k in self.style_enc.state_dict():
                tmp_dict[k] = v
        self.style_enc.load_state_dict(tmp_dict)
        # frozen sty_enc
        for p in self.style_enc.parameters():
            p.requires_grad = False
        
        self.style_enc.to(self.device)

    def korean_index_to_uni_diff(self, indexs : list):
        char_list = []
        for index in indexs:
            unicode_diff = ord(self.dataset_classes[index]) - 44032# ord('가')
            char_list.append(unicode_diff)
        return char_list

    # def set_charAttr_dim(mode):
    #     pass
    def make_condition(self, images, indexs, mask=None):
        input_length = images.shape[0]
        # contents_index = [int(content_index) for content_index in contents_index]
        # style_encoder = style_enc_builder(1,32).to(self.device)
        label = None
        stroke =  None
        style = None
        style_c = 128
        style_h, style_w = 16, 16
        
        contents_p, stroke_p = random.random(), random.random()

        # content
        uni_diff_list = torch.LongTensor(self.korean_index_to_uni_diff(indexs))
        label = torch.FloatTensor(self.contents_emb(uni_diff_list))
        label = label.to(self.device)

        # stroke
        stroke = torch.FloatTensor(self.korean_stroke_emb.embedding(indexs))
        stroke = stroke.to(self.device)

        # style
        style = self.style_enc(images)
        style = style.to(self.device)

        # cal mask
        if mask is not None:
            mask = mask.unsqueeze(dim=-1).to(self.device)
            label = label * mask
            stroke = stroke * mask
            style = style * mask


        condition_dict = {}
        condition_dict["label"] = label
        condition_dict["stroke"] = stroke
        condition_dict['style'] = style

        return condition_dict
    
    