import os
import numpy as np
import math
import random

from PIL import Image
from torch.utils.data import Dataset


def _list_image_files_recursively(data_path):
    result = []
    for entry in sorted(os.listdir(data_path)):
        full_path = f"{data_path}/{entry}"
        ext = entry.split(".")[-1] # 확장자

        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            result.append(full_path)
        elif os.path.isdir(full_path):
            result.extend(_list_image_files_recursively(full_path))
    
    return result


class FontDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 image_size: int,
                 transform = None,
                 ):
        super().__init__()
        if not data_path:
            raise ValueError("data path not exist.")
        self.data_path = data_path
        self.image_size = image_size
        self.transform = transform
        
        # find all files path
        all_files = _list_image_files_recursively(data_path)

        # define class
        class_names = [os.path.basename(path).split("_")[-1].split('.')[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        char_classes = sorted(list(set(class_names)))

        # define sty_class 
        sty_class_names = [os.path.dirname(path).split("/")[-1] for path in all_files]
        sty_sorted_classes = {x: i for i, x in enumerate(sorted(set(sty_class_names)))}
        sty_classes = [sty_sorted_classes[x] for x in sty_class_names]
        
        self.classes = char_classes

        self.local_images = all_files
        self.local_classes = classes
        self.sty_classes = sty_classes

        sty_classes_set = set(self.sty_classes)
        self.sty_classes_idxs = {}
        for sty_class in sty_classes_set:
            self.sty_classes_idxs[sty_class] = np.where(np.array(self.sty_classes) == sty_class)[0]

    def __len__(self):
        return len(self.local_images)
    
    def __getitem__(self, index):
        path = self.local_images[index]
        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.local_classes is not None:
            label = np.array(self.local_classes[index], dtype=np.int64)
        
        if self.sty_classes is not None:
            # image random choice to input sty encoder
            sty_img_idxs = self.sty_classes_idxs[self.sty_classes[index]]
            rand_sty_idx = np.random.choice(sty_img_idxs)
            sty_img_path = self.local_images[rand_sty_idx]
            # load image for sty
            with open(sty_img_path, "rb") as f:
                sty_pil_image = Image.open(f)
                sty_pil_image.load()
            sty_pil_image = sty_pil_image.convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
            sty_pil_image = self.transform(sty_pil_image)

        return label, pil_image, sty_pil_image

    


