import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
# Get Font image
font2 = ImageFont.truetype(font='/root/paper_project/Tools/MakeTTF/font.ttf', size=64)
unicodeChars = 'a'
image_size = 128
# Get font image size and bbox
x,y = font2.getsize(unicodeChars)

print(x,y)

left, top, right, bottom = font2.getbbox(unicodeChars)

print(left, top, right, bottom)

# Check font image is empty / If font image is empty, do not create image
# if x == 0 or y == 0 or (right-left) == 0 or (bottom-top) == 0:
#     continue

# make base image
font_image = Image.new('RGB', (image_size, image_size), color='white')

# Draw font image on base image
draw_image = ImageDraw.Draw(font_image)
draw_image.text(((image_size-x)/2, (image_size-y)/2), unicodeChars[0], font=font2, fill='black')

# Set file name
# file_name = os.path.join(result_path, ttf[:-4]+"_"+unicodeChars)

# print(ttf[:-4],ttf_name)
file_name = '/root/paper_project/Tools/MakeTTF/test'

# Save image
font_image.save('{}.png'.format(file_name)) 