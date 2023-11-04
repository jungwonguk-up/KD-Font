import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
import _thread
import numpy as np
image_size = 64

def make_font_image(font,unicodeChars,file_path):
    # Get font image size and bbox
    x,y = font.getsize(unicodeChars)
    left, top, right, bottom = font.getbbox(unicodeChars)

    # Check font image is empty / If font image is empty, do not create image
    if x == 0 or y == 0 or (right-left) == 0 or (bottom-top) == 0:
        return False

    # Make base image
    font_image = Image.new('RGB', (image_size, image_size), color='white')

    # Draw font image on base image
    draw_image = ImageDraw.Draw(font_image)
    draw_image.text(((image_size-x)/2, (image_size-y)/2), unicodeChars[0], font=font, fill='black')
    
    # Save image
    font_image.save(file_path)
    
    return True
    
        
def make_font_grayscale_image(font,unicodeChars,file_path):
    # Get font image size and bbox
    x,y = font.getsize(unicodeChars)
    left, top, right, bottom = font.getbbox(unicodeChars)

    # Check font image is empty / If font image is empty, do not create image
    if x == 0 or y == 0 or (right-left) == 0 or (bottom-top) == 0:
        return False

    # Make base image
    font_image = Image.new('RGB', (image_size, image_size), color='white')

    # Draw font image on base image
    draw_image = ImageDraw.Draw(font_image)
    draw_image.text(((image_size-x)/2, (image_size-y)/2), unicodeChars[0], font=font, fill='black')
    
    # Convert the image to grayscale
    font_image = ImageOps.grayscale(font_image)
    
    # Save image
    font_image.save(file_path)
    
    return True

if __name__ == '__main__':
    # set parameter
    fonts_base_path = "/usr/share/fonts/truetype"
    result_path = "/home/hojun/Documents/code/cr_diffusion/KD-Font/Tools/MakeFont/Hangul_Characters_Image64_Grayscale"
    csv_path = "/home/hojun/Documents/code/cr_diffusion/KD-Font/Tools/MakeFont/"
    
    if os.path.isdir(result_path):
        pass
    else:
        os.mkdir(result_path)
    
    # make list of font folder names
    fonts_folder_list = os.listdir(fonts_base_path)
    fonts_folder_list = [x for x in fonts_folder_list if x[0] != "."]

    # set making char point(unicode)
    makeing_chars = "괴그기깅나는늘다도디러로를만버없에우워을자점하한했"
    hangul_codePoint = [format(ord(ch),'X') for ch in makeing_chars]

    # Set font's size
    font_size = int(image_size *0.8)
    
    train_files = []
    
    # Generate each character's image with different font
    for fonts_folder in fonts_folder_list:
        fonts = os.listdir(os.path.join(fonts_base_path,fonts_folder))
        fonts = [x for x in fonts if x[0] != "."]
        for ttf in fonts:
            font = ImageFont.truetype(font=os.path.join(fonts_base_path, fonts_folder, ttf), size=font_size)
            
            # Set file name
            ttf_name = ttf[:-4].replace(" ","")
            for uni in tqdm(hangul_codePoint):
                unicodeChars = chr(int(uni, 16))
                # Set file name and path
                file_name = ttf_name + "_" + unicodeChars + ".png"
                file_path = os.path.join(result_path, file_name)
                
                # Make Font Image
                # _thread.start_new_thread(make_font_grayscale_image,(fonts_base_path,fonts_folder,font_size,unicodeChars,result_path))
                make_flag = make_font_grayscale_image(font=font, unicodeChars=unicodeChars, file_path=file_path)
                if make_flag:
                    train_files.append([file_name,file_path,unicodeChars])
    train_csv = pd.DataFrame(train_files)
    train_csv.to_csv(os.path.join(csv_path,"diffusion_font_train.csv"),index=False)
