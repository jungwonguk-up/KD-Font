from PIL import Image,ImageFont,ImageDraw

def make_example_from_ttf(text: str,
                          background_image_path: str,
                          ttf_file_path: str,
                          text_size: int = 50,
                          text_align: str = "center",
                          draw_coordinate: tuple = (180, 50)
                          ) -> Image.Image:
    """
    make example image using ttf file
    """
    
    image = Image.open(background_image_path)
    selected_font = ImageFont.truetype(ttf_file_path, text_size)

    draw = ImageDraw.Draw(image)
    draw.text(draw_coordinate, text, font=selected_font, align=text_align)
    # to do
    path = '/home/wonguk/coding/makettf/example_img'
    image.save(f'{path}.jpg', format='jpeg', quality=100, subsampling=0)



    return image