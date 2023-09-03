import random
import torch
import torch.nn as nn
from modules.style_encoder import style_enc_builder

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
        self.contents_dim = 100 
        self.contents_emb = nn.Embedding(num_classes, self.contents_dim)
        self.korean_stroke_emb = Korean_StrokeEmbedding(txt_path=stroke_text_path,classes=self.dataset_classes)
        self.style_enc = self.make_style_enc(style_enc_path)
        self.language = language

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
        # contents_index = [int(content_index) for content_index in contents_index]
        # style_encoder = style_enc_builder(1,32).to(self.device)
        contents = None
        stroke =  None
        style = None
        contents_p, stroke_p = random.random(), random.random()

        style = self.style_enc(images)
        b, c, h, w = style.shape

        if mode == 1:
            if contents_p < 0.3:
                contents = torch.zeros(input_length, self.contents_dim)
            else:
                uni_diff_list = torch.LongTensor(self.korean_index_to_uni_diff(indexs))
                contents = torch.FloatTensor(self.contents_emb(uni_diff_list))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length, 68)
            else:
                stroke =  torch.FloatTensor(self.korean_stroke_emb.embedding(indexs))
            
            if contents_p < 0.3 and stroke_p < 0.3:
                style = torch.zeros(b, c, h*w).cpu()

            else:
                # style = self.style_enc(images).flatten(1).cpu()
                style = style.view(b, c, -1).cpu()

        elif mode == 2:
            if contents_p < 0.3:
                contents = torch.zeros(input_length, self.contents_dim)
            else:
                contents = torch.FloatTensor(self.korean_index_to_uni_diff(indexs))

            if stroke_p < 0.3:
                stroke = torch.zeros(input_length, 68)
            else:
                stroke =  torch.FloatTensor(self.korean_stroke_emb.embedding(indexs))
            
            # style = torch.zeros(input_length,32768).cpu()
            style = torch.zeros(b, c, h*w).cpu()


        elif mode == 3: #test
            uni_diff_list = torch.LongTensor(self.korean_index_to_uni_diff(indexs))
            contents = torch.FloatTensor(self.contents_emb(uni_diff_list))
            stroke =  torch.FloatTensor(self.korean_stroke_emb.embedding(indexs))
            # style = self.style_enc(images).flatten(1).cpu()
            style = style.view(b, c, -1).cpu()

        # return torch.cat([contents,stroke,style],dim=1)
        return style, contents, stroke
    
    