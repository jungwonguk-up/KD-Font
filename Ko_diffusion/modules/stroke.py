import torch
import torch.nn as nn


class StrokeEmbedding():
    def __init__(self,txt_path):
        self.emd_stroke_list = self.read_stroke_txt(txt_path)

    def read_stroke_txt(self, txt_path):
        emd_stroke_list = []
        read_txt = open(txt_path, 'r')
        for emd in read_txt:
            emd_stroke_list.append(emd[:-1])

        return emd_stroke_list

    def embedding(self, label):
        stroke_embedding = []
        for stroke in label:
            tmp = []
            for check in self.emd_stroke_list[stroke]:
                for i in check:
                    tmp.append(int(i))
            stroke_embedding.append(tmp)
        stroke_embedding = torch.Tensor(stroke_embedding)
        return stroke_embedding


