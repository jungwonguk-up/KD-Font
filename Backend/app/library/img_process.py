import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageFont
import numpy as np
import imutils


class PreProcess:

    @staticmethod
    def PIL2CV(image: Image.Image) -> np.array:
        """convert pillow image to opencv image, change color RGB -> BGR"""
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def CV2PIL(image: np.array) -> Image.Image:
        """convert opencv image to pillow image, change color BGR -> RGB"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    @staticmethod
    def CV_resize(image: np.array, resize_width: int = 1200) -> np.array:
        h, w, _ = image.shape
        resize_ratio = resize_width / w # resize_width / original_width
        resize_img = cv2.resize(image, dsize=(0,0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
        return resize_img
    
    @staticmethod
    def PIL_resize(image: Image.Image, size: int = 200) -> Image.Image:
        return image.resize((size, size), Image.Resampling.LANCZOS)
    
    @staticmethod
    def PIL_crop(image: Image.Image, crop_size: int = 10) -> Image.Image:
        h, w = image.size
        return image.crop((crop_size, crop_size, h-crop_size, w-crop_size))

    @staticmethod
    def PIL_adjust_contrast(image: Image.Image, enhance_value: float = 1.) -> Image.Image:
        """
        Additional contrast adjustment for pillow image
        image (pillow image): input image
        enhance_value (int): Determine contrast enhance level
        """
        contrast_img = ImageEnhance.Contrast(image)
        return contrast_img.enhance(enhance_value)
    
    @staticmethod
    def PIL_smoothing_edge(image: Image.Image, filter_size: int = 5) -> Image.Image:
        """Smoothing alphabet letter edge"""
        return image.filter(ImageFilter.ModeFilter(size=filter_size))
    
    @staticmethod
    def detect_edge(image: np.array) -> np.array:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # Edge Detection.
        canny = cv2.Canny(gray, 0, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return canny
    
    @staticmethod
    def detect_contour(canny: np.array) -> list:
        # Finding contours for the detected edges.
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # Keeping only the largest detected contour.
        return contours

    @staticmethod
    def find_retangle_corners(contours: list) -> list:
        """Find biggest retangle corners from contours"""
        # Loop over the contours.
        for c in contours:
            epsilon = 0.02 * cv2.arcLength(c, True) # Approximate the contour.
            corners = cv2.approxPolyDP(c, epsilon, True)
            if len(corners) == 4: # If our approximated contour has four points
                break

        corners = sorted(np.concatenate(corners).tolist()) # Sorting the corners and converting them to desired shape.
        return corners
    
    @staticmethod
    def perspective_transform(image: np.array, corners: list, size: int = 400) -> np.array:
        """perspective transform by corners"""
        pts1 = np.float32(corners)
        pts2 = np.float32([[0, 0], [0, size], [size, 0], [size, size]])

        m = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, m, (size, size))
        return dst
    
    @staticmethod
    def morphology(image: np.array, kernel_shape: str = 'ellipse', kernel_size: tuple = (5, 5)):
        assert kernel_shape.lower() == "ellipse" or "rect" or "cross"
        if kernel_shape.lower() == "ellipse":
            shape = cv2.MORPH_ELLIPSE
        elif kernel_shape.lower() == "rect":
            shape = cv2.MORPH_RECT
        else:
            shape = cv2.MORPH_CROSS

        kernel = cv2.getStructuringElement(shape, kernel_size)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return opening
    
    @staticmethod
    def automatic_brightness_and_contrast(image: np.array,
                                          alpha_mul: int = 1.,
                                          beta_mul: int = 1.,
                                          clip_hist_percent: int = 1
                                          ) -> np.array:
        """
        adjust image brightness and contrast automaticully.
        image (numpy array): input image 
        alpha_mul (int): alpha(brightness) multiply param to adjust background color white. default=1.5 (experiment value)
        beta_mul (int): beta(contrast) multiply param to adjust letter more vivid. default=1.5 (experiment value)
        clip_hist_percent (int): cumulative distribution to determine where color frequency is less than threshold value. default=1(%)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
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
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha*alpha_mul, beta=beta*beta_mul)
        return auto_result

    

def image_processing(image: Image.Image,
                     size: int = 200,
                     edge_crop_size: int = 10,
                     brightness_adj: float = 1.5,
                     contrast_enhance: float = 2,
                     inner_resize_w: int = 1200
                     ) -> Image.Image:
    """
    Image Pre-Processing Function.
    """
    image = image.convert("L")
    image = PreProcess.PIL2CV(image)
    image = PreProcess.CV_resize(image, resize_width=inner_resize_w)

    # detect retangle and find retangle corner
    canny = PreProcess.detect_edge(image)
    contour = PreProcess.detect_contour(canny)
    corner = PreProcess.find_retangle_corners(contour)

    # perspective_transform
    image = PreProcess.perspective_transform(image, corner, size*2)

    #adjust image morp and color
    image = PreProcess.automatic_brightness_and_contrast(image, alpha_mul=brightness_adj)
    image = PreProcess.morphology(image, kernel_shape='ellipse', kernel_size=(5, 5))

    # convert cv image to pillow and adjust image color 
    image = PreProcess.CV2PIL(image)
    image = PreProcess.PIL_adjust_contrast(image, enhance_value=contrast_enhance)
    image = PreProcess.PIL_smoothing_edge(image, filter_size=5)

    # crop edge and resize
    image = PreProcess.PIL_crop(image, crop_size=edge_crop_size)
    image = PreProcess.PIL_resize(image, size=size)
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
