import os
import shutil

def make_ocr_dataset(base_path = "./data/Hangul_Characters_Image64",to_file_path = "./data/Hangul_Characters_Image64_OCR/test",gt_path = "./data/Hangul_Characters_Image64_OCR/gt.txt"):

    os.makedirs(to_file_path, exist_ok=True)
    image_path = []
    folder_list = os.listdir(base_path)
    f = open(gt_path, 'w')
    for folder_name in folder_list:
        folder_image_path_list = []
        for image_name in os.listdir(os.path.join(base_path, folder_name)):
            f.write(os.path.join("test", image_name) + "\t" + folder_name + "\n")
            shutil.copyfile(os.path.join(base_path, folder_name, image_name), os.path.join(to_file_path, image_name))

    f.close()


make_ocr_dataset()