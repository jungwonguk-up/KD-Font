import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
        def __init__(self, csv_path, transform =None):
            self.transform = transform
            self.csv_data = pd.read_csv(csv_path)
            self.x_file_name = self.csv_data.iloc[:,0]
            self.x_path = self.csv_data.iloc[:,1]
            self.y_chs = self.csv_data.iloc[:,2]
            self.labels = np.unique(self.y_chs)
            self.y_to_label = self.make_y_to_label()
            self.label_to_y = self.make_label_to_y()
            self.y_labels = self.make_y_labels()

        def make_y_to_label(self):
            y_to_label_dict = {}
            for label, value in enumerate(self.labels):
                y_to_label_dict[value] = label
            return y_to_label_dict
        
        def make_label_to_y(self):
            label_to_y_dict = {}
            for label, value in enumerate(self.labels):
                label_to_y_dict[label] = value
            return label_to_y_dict
        
        def make_y_labels(self):
            y_labels = []
            for y_ch in self.y_chs:
                y_labels.append(self.y_to_label[y_ch])
            return y_labels
        
        def __len__(self):
            return len(self.x_path)
        
        def __getitem__(self, id_: int):
            filename = self.x_file_name[id_]
            x = Image.open(self.x_path[id_])
            if self.transform is not None:
                transform_x = self.transform(x)
            y_ch = self.y_chs[id_]
            
            return transform_x, y_ch, filename
        
class DiffusionSamplingDataset(Dataset):
    def __init__(self,sampling_img_path, sampling_chars,img_size, device,transforms):
        self.sampling_chars = sampling_chars
        self.img_size = img_size
        self.sampling_img_path = sampling_img_path
        self.transforms = transforms
        self.device = device
        self.sample_img = self.transforms(Image.open(sampling_img_path))
        self.all_x = np.random.randn(len(sampling_chars), 1, self.img_size, self.img_size)
    
    def __len__(self):
        return len(self.sampling_chars)
    
    def __getitem__(self, id_: int):
        # sample_img = self.transforms(Image.open(self.sampling_img_path))
        sample_x = self.all_x[id_]
        sample_y = self.sampling_chars[id_]
        return self.sample_img,sample_x, sample_y