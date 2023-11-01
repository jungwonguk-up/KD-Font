import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imutils
from matplotlib import pyplot as plt


async def image_preprocess(image: Image.Image, 
                           return_img_size: int = 400,
                           black_threshold: int = 150,
                           inner_resize_width: int = 1200,
                           ) -> Image.Image:
    """
    Detect Retangle and PerciveTransform Processing Function
    image (pillow Image): input image
    return_img_size (int): return image size, default = 400,
    black_threshold (int): Treshold value to make black and white image
    inner_resize_width (int): inner resize value for detect retangle in image. default = 1200 (experiment value)
    """

    # assert type(image_array)
    size = return_img_size

    # convert pillow image to numpy array
    image_array = np.array(image)
    h, w, _ = image_array.shape
    resize_ratio = inner_resize_width / w
    resize_img = cv2.resize(image_array, dsize=(0,0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)

    # # convert the resize_img to grayscale, blur, and find edges
    gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    doc_cnt = None
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                doc_cnt = approx
                break
    
    # perspective transform
    pts1 = np.float32([doc_cnt[0][0], doc_cnt[1][0], doc_cnt[3][0], doc_cnt[2][0]])
    pts2 = np.float32([[0, 0], [0, size], [size, 0], [size, size]])

    m = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(resize_img, m, (size, size))

    # convert array to pillow image
    image = Image.fromarray(dst)

    fn = lambda x : 255 if x > black_threshold else 0
    bw_image = image.convert('L').point(fn, mode='1')

    # TODO 테두리 제거, 이미지 회전

    return bw_image


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

    return image

