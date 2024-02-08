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
    # 모든 contents 이름 바꾸기
    def __init__(self, num_classes, stroke_text_path, style_enc_path, data_classes, language, device):
        self.device = device
        self.dataset_classes = data_classes
        self.num_classes = num_classes
        self.contents_dim = 60 
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

    def korean_index_to_uni_diff(self, indexs : list):
        char_list = []
        for index in indexs:
            unicode_diff = ord(self.dataset_classes[index]) - 44032# ord('가')
            char_list.append(unicode_diff)
        return char_list

    # def set_charAttr_dim(mode):
    #     pass
    def make_condition(self, images, indexs, mode):
        input_length = images.shape[0]
        # make channel 1 to 3 to input style enc (b, 1, h, w) -> (b, 3, h, w)
        images = images.expand(-1, 3 -1, -1)
        # contents_index = [int(content_index) for content_index in contents_index]
        # style_encoder = style_enc_builder(1,32).to(self.device)
        contents = None
        stroke =  None
        style = None
        style_c = 128
        style_h, style_w = 16, 16
        
        contents_p, stroke_p = random.random(), random.random()
        if mode == 1:
            if contents_p < 0.3:
                contents = torch.zeros(input_length,self.contents_dim)
            else:
                uni_diff_list = torch.LongTensor(self.korean_index_to_uni_diff(indexs))
                contents = torch.FloatTensor(self.contents_emb(uni_diff_list))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length,68)
            else:
                stroke =  torch.FloatTensor(self.korean_stroke_emb.embedding(indexs))
            
            if contents_p < 0.3 and stroke_p < 0.3:
                style = torch.zeros(input_length,style_c, style_h , style_w)
            else:
                style = self.style_enc(images).cpu()
                # style = style.view(input_length, style_c, -1).cpu()
        elif mode == 2:
            if contents_p < 0.3:
                contents = torch.zeros(input_length,self.contents_dim)
            else:
                contents = torch.FloatTensor(self.korean_index_to_uni_diff(indexs))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length,68)
            else:
                stroke =  torch.FloatTensor(self.korean_stroke_emb.embedding(indexs))
            
            style = torch.zeros(input_length,style_c, style_h , style_w)


        elif mode == 3: #test
            uni_diff_list = torch.LongTensor(self.korean_index_to_uni_diff(indexs))
            contents = torch.FloatTensor(self.contents_emb(uni_diff_list))
            stroke =  torch.FloatTensor(self.korean_stroke_emb.embedding(indexs))
            style = self.style_enc(images).cpu()
            
            # style = style.view(input_length, style_c, -1).cpu()
        condition_dict = {}
        condition_dict["contents"] = contents.to(self.device)
        condition_dict["stroke"] = stroke.to(self.device)
        condition_dict['style'] = style.to(self.device)

        return condition_dict
    
    