import fontforge

font = fontforge.font()
font.encoding = 'UnicodeFull'
charactor = '가'
output_path = '/root/paper_project/Tools/MakeTTF/font.ttf'
image_path = '/root/paper_project/Tools/MakeTTF/d03fc0a9c3190dce.png'
image_glyph = font.createChar(ord(charactor), charactor)
image_glyph.importOutlines(image_path)
font.generate(output_path)