import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import _thread

image_size = 64

def make_font_image(fonts_base_path,fonts_folder,font_size,unicodeChars,result_path):
    for ttf in fonts:
        # Get Font image
        font = ImageFont.truetype(font=os.path.join(fonts_base_path, fonts_folder, ttf), size=font_size)

        # Get font image size and bbox
        x,y = font.getsize(unicodeChars)
        left, top, right, bottom = font.getbbox(unicodeChars)

        # Check font image is empty / If font image is empty, do not create image
        if x == 0 or y == 0 or (right-left) == 0 or (bottom-top) == 0:
            continue

        # make base image
        font_image = Image.new('RGB', (image_size, image_size), color='white')

        # Draw font image on base image
        draw_image = ImageDraw.Draw(font_image)
        draw_image.text(((image_size-x)/2, (image_size-y)/2), unicodeChars[0], font=font, fill='black')

        # Set file name
        # file_name = os.path.join(result_path, ttf[:-4]+"_"+unicodeChars)
        file_name = result_path + "/" + ttf[:-4] + "_" + unicodeChars

        # Save image
        font_image.save('{}.png'.format(file_name))

if __name__ == '__main__':
    # set base path
    fonts_base_path = "/usr/share/fonts/truetype"

    # make list of font folder names
    fonts_folder_list = os.listdir(fonts_base_path)
    fonts_folder_list = [x for x in fonts_folder_list if x[0] != "."]

    # set start and end code point(unicode)
    start = "AC00"
    end = "D7AF"

    # hangul syllables's code point list
    hangul_codePoint = list(range(int(start, 16), int(end, 16) + 1))
    hangul_codePoint = [format(x, 'X') for x in hangul_codePoint]

    # Set font's size
    font_size = int(image_size *0.8)

    # Generate each character's folder
    for uni in tqdm(hangul_codePoint):
        unicodeChars = chr(int(uni, 16))

        result_path = "/home/hojun/PycharmProjects/diffusion_font/code/make_font/Hangul_Characters_Image64/" + unicodeChars

        os.makedirs(result_path, exist_ok=True)

    # Generate each character's image with different font
    for uni in tqdm(hangul_codePoint):
        unicodeChars = chr(int(uni, 16))
        result_path = "/home/hojun/PycharmProjects/diffusion_font/code/make_font/Hangul_Characters_Image64/" + unicodeChars

        for fonts_folder in fonts_folder_list:
            fonts = os.listdir(os.path.join(fonts_base_path,fonts_folder))
            fonts = [x for x in fonts if x[0] != "."]
            # _thread.start_new_thread(make_font_image,(fonts_base_path,fonts_folder,font_size,unicodeChars,result_path))
            make_font_image(fonts_base_path,fonts_folder,font_size,unicodeChars,result_path)