import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path: str) -> Image.Image:
    """open image file as PIL Image"""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def list_image_files_recursively(data_path: str) -> list:
    """find all files path and return to list"""
    result = []
    for entry in sorted(os.listdir(data_path)):
        full_path = f"{data_path}/{entry}"
        ext = entry.split(".")[-1] # 확장자

        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            result.append(full_path)
        elif os.path.isdir(full_path):
            result.extend(list_image_files_recursively(full_path))
    
    return result


class FontDataset(Dataset):
    """
    Dataset for data directory sorted by font class folders.

    dataset structured as follows:

    directory/
    ├── font1
    │   ├── font1_a.png
    │   ├── font1_b.png
    │   └── ...
    │    
    └── font2
        ├── font2_a.png
        ├── font2_b.png
        └── ...
        └── font2_z.png
    ...

    """
    def __init__(self,
                 data_path: str,
                 image_size: int = None,
                 cfg_mode: int = 1,
                 transform = None,
                 sty_transform = None,
                 ):
        super().__init__()
        if not data_path:
            raise ValueError("data path not exist.")
        self.data_path = data_path
        self.image_size = image_size # not use currently
        self.cfg_mode = cfg_mode
        self.transform = transform
        self.sty_transform = sty_transform
        
        # find all files path
        all_files = list_image_files_recursively(data_path)

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
        pil_image = pil_loader(path)

        if self.local_classes is not None:
            label = np.array(self.local_classes[index], dtype=np.int64)
        
        if self.sty_classes is not None:
            # image random choice to input sty encoder
            sty_img_idxs = self.sty_classes_idxs[self.sty_classes[index]]
            rand_sty_idx = np.random.choice(sty_img_idxs)
            sty_img_path = self.local_images[rand_sty_idx]
            # load image for sty
            sty_pil_image = pil_loader(sty_img_path)

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.sty_transform is not None:
            sty_pil_image = self.sty_transform(sty_pil_image)

        label_p, stroke_p, style_p = np.random.random(), np.random.random(), np.random.random()

        mask = True

        if self.cfg_mode == 1:
            if label_p < 0.1:
                mask = False
        
        return label, pil_image, sty_pil_image, mask


class CharDataset(Dataset):
    """
    Dataset for data directory sorted by Charater class folders.

    dataset structured as follows:

    directory/
    ├── a
    │   ├── font1_a.png
    │   ├── font2_a.png
    │   └── ...
    │    
    └── b
        ├── font1_b.png
        ├── font2_b.png
        └── ...
        └── font100_b.png
    ...

    """
    def __init__(self,
                 data_path: str,
                 image_size: int = None,
                 cfg_mode: int = 1,
                 transform = None,
                 sty_transform = None,
                 ):
        super().__init__()
        if not data_path:
            raise ValueError("data path not exist.")
        self.data_path = data_path
        self.image_size = image_size # not use currently
        self.cfg_mode = cfg_mode
        self.transform = transform
        self.sty_transform = sty_transform
        
        # find all files path
        all_files = list_image_files_recursively(data_path)

        # define class
        class_names = [os.path.dirname(path).split("/")[-1] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        char_classes = sorted(list(set(class_names)))

        # define sty_class 
        sty_class_names = ["_".join(os.path.basename(path).split("_")[:-1]) for path in all_files]
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
        pil_image = pil_loader(path)

        if self.local_classes is not None:
            label = np.array(self.local_classes[index], dtype=np.int64)
        
        if self.sty_classes is not None:
            # image random choice to input sty encoder
            sty_img_idxs = self.sty_classes_idxs[self.sty_classes[index]]
            rand_sty_idx = np.random.choice(sty_img_idxs)
            sty_img_path = self.local_images[rand_sty_idx]
            # load image for sty
            sty_pil_image = pil_loader(sty_img_path)

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.sty_transform is not None:
            sty_pil_image = self.sty_transform(sty_pil_image)

        label_p, stroke_p, style_p = np.random.random(), np.random.random(), np.random.random()

        mask = True

        if self.cfg_mode == 1:
            if label_p < 0.1:
                mask = False
        

        return label, pil_image, sty_pil_image


