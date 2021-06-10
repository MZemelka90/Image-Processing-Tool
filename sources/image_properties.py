import numpy as np
from PIL import ImageEnhance
from skimage.metrics import structural_similarity, mean_squared_error
from skimage import measure


class ImagePreProcessing:

    def __init__(self, img):
        self.img = img

    def contrast(self):
        pass


    def ssim_and_error(self, img1=0, img2=1):
        img_ssim = structural_similarity(self.img[img1], self.img[img2])
        img_error = mean_squared_error(self.img[img1], self.img[img2])
        return img_ssim, img_error

    def crop(self):
        pass

