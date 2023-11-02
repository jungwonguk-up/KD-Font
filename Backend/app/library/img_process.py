import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import numpy as np
import imutils


class PreProcessing:

    @staticmethod
    def convert_PIL2CV(image: Image.Image) -> np.array:
        """convert pillow image to opencv image, change color RGB -> BGR"""
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def convert_CV2PIL(image: np.array) -> Image.Image:
        """convert opencv image to pillow image, change color BGR -> RGB"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    @staticmethod
    def PIL_resize(image: Image.Image, size: tuple = (200, 200)) -> Image.Image:
        return image.resize(size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def PIL_crop(image: Image.Image, crop_size: int = 10) -> Image.Image:
        h, w = image.size
        return image.crop((crop_size, crop_size, h-crop_size, w-crop_size))
    
    @staticmethod
    def PIL_adjust_contrast(image: Image.Image, enhance_value: float = 1.) -> Image.Image:
        """
        Additional contrast adjustment for pillow image
        image (pillow image): input image
        enhance_value : Determine contrast enhance level
        """
        contrast_img = ImageEnhance.Contrast(image)
        return contrast_img.enhance(enhance_value)

    @staticmethod
    def automatic_brightness_and_contrast(image: np.array,
                                          alpha_mul: float = 1.,
                                          beta_mul: float = 1.,
                                          clip_hist_percent: int = 1
                                          ) -> np.array:
        """
        Automatically adjust image brightness and contrast.
        alpha_mul (float): alpha(brightness) multiply param to adjust background color white. higher value make image more bright. default=1.
        beta_mul (float): beta(contrast) multiply param to adjust letter more vivid. higher value make image more contrast. defulat=1.
        clip_hist_percent (int): cumulative distribution to determine where color frequency is less than threshold value. default=1(%)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = (-minimum_gray * alpha) * beta_mul
        alpha = alpha * alpha_mul

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return auto_result

    @staticmethod
    def perspective_transform(image: np.array, 
                              return_img_size: int = 200, 
                              inner_resize_width: int = 1200,
                              ) -> np.array:
        """
        Detect Retangle and PerciveTransform Processing.
        image (pillow Image): input image
        return_img_size (int): return image size
        inner_resize_width (int): inner resize value for detect retangle in image. default = 1200 (experiment value)
        """
        # assert type(image_array)
        size = return_img_size
        
        h, w, _ = image.shape
        resize_ratio = inner_resize_width / w
        resize_img = cv2.resize(image, dsize=(0,0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)

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
        return dst
    

def pre_processing(image: Image.Image,
                   size: int = (200, 200),
                   edge_crop_size: int = 10,
                   brightness_adj_val: float = 1.,
                   contrast_adj_val: float = 1.,
                   contrast_enhance_val: float = 1.,
                   ) -> Image.Image:
    """
    Image Pre-Processing Function.
    """
    image = PreProcessing.convert_PIL2CV(image)
    image = PreProcessing.perspective_transform(image, return_img_size=size[0]*2, inner_resize_width=1200)
    image = PreProcessing.automatic_brightness_and_contrast(image, alpha_mul=brightness_adj_val, beta_mul=contrast_adj_val)
    image = PreProcessing.convert_CV2PIL(image)
    image = PreProcessing.PIL_adjust_contrast(image, enhance_value=contrast_enhance_val)
    image = PreProcessing.PIL_crop(image, crop_size=edge_crop_size)
    image = PreProcessing.PIL_resize(image, size=size)
    return image


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
