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
    result_path = "/home/hojun/Documents/code/cr_diffusion/KD-Font/Data/Hangul_Characters_Image64_Grayscale"
    csv_path = "/home/hojun/Documents/code/cr_diffusion/KD-Font/Data"
    
    if os.path.isdir(result_path):
        pass
    else:
        os.mkdir(result_path)
    
    # make list of font folder names
    fonts_folder_list = os.listdir(fonts_base_path)
    fonts_folder_list = [x for x in fonts_folder_list if x[0] != "."]

    # set making char point(unicode)
    makeing_chars = "쏘긺펄멫늛훤쯆틃떖겮퐼윕쓃퐘행찺훻핊퀒트죋볣텝릤웜뽺퓀깍쎹죵홀좺줽즡쾨킵츒횛췕뚚싩넉멤쵩굍몍뽏킲뤦쒮삈뻜믗쓓큣젅몣쏡쓀뿣넄쌬핶퍂됺룚픐췚텲율풴췜두뮞꽎쪡꺇뚞쯗쩁햜덎잠뉡좀뵿뺶즏갵윌긛뵞핵락춲쥅껳긂렃돞뺚춗쉢광촌웸걲쉍삚뒡섈셆왥쁢몆뵭럛죠냃좋졇찣햯쏠냻밡슫쭩겾럕퇝쓘혌츀줳쪑걹씶옢킫횦칇샱뚏킞홦딹쮜숕봥닶퇌늍콅뎣쇕쪳찄뎭똉털펚믩쾤툖횭뢗듛엻튄꺀줺댞쿠땡쯰푽콟갌뢧퍪틟킖씿넲횆귚뿌췦퀩뺛놝쓠뭹얔펎뢥콕큫깎엨뺂늺뒲꼫깇뽮퓲뇉꾓쏉뵇씗렱묙횃쎈똌욣틞턀뷛엉넪쮚턿덳뒤균봾퇐묵돶쪰퓤봒쯘깄횽븇웢쇃덞겼쬛몐깬굄혲펯숚떐햓슲첊텕뼯뢀릳촢츄딿쩞욥뒠웤탇헂큸삡팫챗츻죍떟떂셨햱꿚빽곅췵병꾇떥됪뙽쵫작옛톥뮸벙쭰녺쓜졎꿥밙껪챣릗뱅욨떰쟒즲럲쁘웅톿럊갊꾏롯꿉꺞쭎믭엝푶큓삤벳뭫깮받왃웬렪쓽햍뙧뙚쏸퀔퐵퀖톡땸쀬욬꽝껏벪킏믧팄땂숋뽽껕욍쓫쯄먫뚤뗹핕쵹난묹돬첍틨싵첃뫌닒뽍밭즟곭퓐됂땪떀눷쫅갔댈샛븋묏무뎕붧캻줒캼쪍몊꿗캴좌꽤퓦붿쒇픏섔슰뭈뎀꿍텐긟뒮퉃룻뻭팱셚홋헔굀싦쪥븃띾괴그기깅나는늘다도디러로를만버없에우워을자점하한했"
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
