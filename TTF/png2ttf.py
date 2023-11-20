from PIL import Image, ImageChops
import os
import shutil
import subprocess
import fontforge
import os


class PotraceNotFound(Exception):
    pass

class PNGtoSVG:
    def __init__(self, directory):
        self.directory = directory

    def convert(self):
        path = os.walk(self.directory)

        for root, dirs, files in path:
            for f in files:
                if f.endswith(".png"):
                    self.pngToBmp(root + "/" + f)
                    self.bmpToSvg(root + "/" + f[0:-4] + ".bmp")

    def bmpToSvg(self, path):
        if shutil.which("potrace") is None:
            raise PotraceNotFound("Potrace is either not installed or not in path")
        else:
            new_name = str(ord(path[-5]))
            subprocess.run(["potrace", path, "-b", "svg", "-o",path[:-5]+ new_name + ".svg"])

    def pngToBmp(self, path):
        img = Image.open(path).convert("RGBA").resize((90, 90)) # 100,100
        threshold = 200
        data = []
        new_name = (path[-5])
        for pix in list(img.getdata()):
            if pix[0] >= threshold and pix[1] >= threshold and pix[3] >= threshold:
                data.append((255, 255, 255, 0))
            else:
                data.append((0, 0, 0, 1))
        img.putdata(data)
        img.save(path[:-5] + new_name + ".bmp")

    def trim(self, im_path):
        im = Image.open(im_path)
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        bbox = list(diff.getbbox())
        bbox[0] -= 1
        bbox[1] -= 1
        bbox[2] += 1
        bbox[3] += 1
        cropped_im = im.crop(bbox)
        cropped_im.save(im_path)


class FontCreator:
    def __init__(self, start_unicode, end_unicode, output_font):
        self.start_unicode = start_unicode
        self.end_unicode = end_unicode
        self.output_font = output_font

    def create_font(self, svg_path):
        font = fontforge.font()

        space = font.createChar(0x20)
        space.width = 500

        for unicode in range(self.start_unicode, self.end_unicode+1):  
            print(unicode)       
            char = font.createChar(unicode) 
            svg_file = os.path.join(svg_path,f"{unicode}.svg")

            try:
                glyph = char.importOutlines(svg_file)

                char.left_side_bearing = 50  # adjust as needed
                char.right_side_bearing = 50
            except:
                pass

        font.generate(self.output_font)



class MakeTTF:
    def __init__(self,path, uuid, ttf_path):
        self.path = path
        self.uuid = uuid
        self.ttf_path = ttf_path

    def create_ttf(self):
        dir_path = os.path.dirname(self.path)
        converter = PNGtoSVG(dir_path)
        converter.convert()
        creator = FontCreator(0xAC00, 0xD7A3, f"{self.ttf_path}/{self.uuid}_sample.ttf")
        creator.create_font(dir_path)
        return f"{self.ttf_path}/{self.uuid}_sample.ttf"
        