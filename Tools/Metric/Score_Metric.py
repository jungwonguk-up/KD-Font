import lpips
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from pytorch_msssim import ssim
import torch
import torch.nn as nn
import math
import statistics

class ImageQualityMetrics:
    def __init__(self, original_dir, generate_dir):
        self.original_dir = original_dir
        self.generate_dir = generate_dir

    @staticmethod
    def psnr_cal(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / torch.sqrt(mse))

    @staticmethod
    def lpips_cal(img1, img2, backborn='alex'):
        assert backborn == 'alex' or 'vgg'

        loss_fn = lpips.LPIPS(net=backborn)
        return loss_fn.forward(img1, img2)

    @staticmethod
    def load_image_from_folder(folder_path):
        valid_image_extenstions = ('.jpg', '.jpeg', '.png', '.bmp')
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_image_extenstions):
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path)
                    image_np = np.array(image)
                    transform = transforms.ToTensor()
                    image_tensor = transform(image_np)
        return image_tensor

    def cal_score(self):
        psnr_scores = []
        lpips_scores = []
        ssim_scores = []
        folders1 = [f for f in os.listdir(self.original_dir) if os.path.isdir(os.path.join(self.original_dir, f))]
        folders2 = [f for f in os.listdir(self.generate_dir) if os.path.isdir(os.path.join(self.generate_dir, f))]

        for folder1, folder2 in zip(folders1, folders2):
            image1 = self.load_image_from_folder(os.path.join(self.original_dir, folder1))
            image2 = self.load_image_from_folder(os.path.join(self.generate_dir, folder2))
            psnr_scores.append(self.psnr_cal(image1, image2))
            lpips_scores.append(self.lpips_cal(image1, image2))
            batch_image1 = image1.unsqueeze(0)
            batch_image2 = image2.unsqueeze(0)
            ssim_scores.append(ssim(batch_image1, batch_image2, data_range=255, size_average=False))

        mean_psnr_score = statistics.mean(psnr_scores)
        float_lpips_scores = [tensor.item() for tensor in lpips_scores]
        mean_lpips_score = statistics.mean(float_lpips_scores)
        float_ssim_scores = [tensor.item() for tensor in ssim_scores]
        mean_ssim_score = statistics.mean(float_ssim_scores)

        return mean_psnr_score, mean_lpips_score, mean_ssim_score

if __name__ == "__main__":
    original_dir = '/root/paper_project/ML/sample_data'
    generate_dir = '/root/paper_project/ML/sample_data'
    metrics_calculator = ImageQualityMetrics(original_dir, generate_dir)
    psnr, lpips, ssim = metrics_calculator.cal_score()
    print("PSNR:", psnr)
    print("LPIPS:", lpips)
    print("SSIM:", ssim)