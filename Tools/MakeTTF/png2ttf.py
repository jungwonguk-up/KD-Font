from PIL import Image, ImageChops
import os
import shutil
import subprocess
import fontforge

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
            subprocess.run(["potrace", path, "-b", "svg", "-o", path[0:-4] + ".svg"])

    def pngToBmp(self, path):
        img = Image.open(path).convert("RGBA").resize((100, 100))
        threshold = 200
        data = []
        for pix in list(img.getdata()):
            if pix[0] >= threshold and pix[1] >= threshold and pix[3] >= threshold:
                data.append((255, 255, 255, 0))
            else:
                data.append((0, 0, 0, 1))
        img.putdata(data)
        img.save(path[0:-4] + ".bmp")

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

    def create_font(self):
        font = fontforge.font()
        for unicode in range(self.start_unicode, self.end_unicode+1):
            char = font.createChar(unicode)
            svg_file = f"img/{unicode}.svg"

            try:
                glyph = char.importOutlines(svg_file)
            except:
                pass

        font.generate(self.output_font)


converter = PNGtoSVG('/img')
converter.convert()

creator = FontCreator(0xAC00, 0xD7A3, "myfont.ttf")
creator.create_font()