import cv2

class SuperResolution:
    def __init__(self, model_path, scale_factor=4):
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        self.sr.setModel("edsr", scale_factor)

    def upscale_image(self, img_path, output_path):
        img = cv2.imread(img_path)
        upscaled_img = self.sr.upsample(img)
        cv2.imwrite(output_path, upscaled_img)

if __name__ == "__main__":
    model_path = 'EDSR_x4.pb'
    scale_factor = 4
    image_path = '/home/wonguk/coding/SuperResolution/횆.png'
    output_path = 'output.png'

    sr = SuperResolution(model_path, scale_factor)
    sr.upscale_image(image_path, output_path)