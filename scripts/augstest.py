import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from PIL import ImageDraw, Image

from skimage import color


class SmartResize:
    """
    Resizes image avoiding changing its aspect ratio.
    Includes stretching and squeezing augmentations.
    """

    def __init__(self, width, height, stretch=(1, 1), fillcolor=0):
        """
        Args:
        
            width (int): target width of the image.
            
            height (int): target height of the image.
            
            stretch (tuple): defaults to (1, 1) - turned off.
            Parameter for squeezing / stretching augmentation,
            tuple containining to values which represents
            the range of queezing / stretching.
            Values less than 1 compress the image from the sides (squeezing),
            values greater than 1 stretch the image to the sides (stretching).
            Use range (1, 1) to avoid squeezing / stretching augmentation.
            
            fillcolor (int): defults to 255 - white. Number in range [0, 255]
            representing fillcolor.
        """

        assert len(
            stretch) == 2, "stretch has to contain only two values " \
                           "representing range. "
        assert stretch[0] >= 0 and stretch[
            1] >= 0, "stretch has to contain only positive values."
        assert 0 <= fillcolor <= 255, "fillcolor has to contain values in " \
                                      "range [0, 255]. "

        self.width = int(width)
        self.height = int(height)
        self.stretch = stretch
        self.ratio = int(width / height)
        self.color = fillcolor

    def __call__(self, img):
        """
        Transformation.
        Args:
            img (np.array): RGB np.array image which has to be transformed.
        """

        stretch = random.uniform(self.stretch[0], self.stretch[1])

        img = np.array(img)
        h, w, _ = img.shape
        img = cv2.resize(img, (w, int(w / (w * stretch / h))))
        h, w, _ = img.shape

        if not (w / h) == self.ratio:
            x = 0.5
            if (w / h) < self.ratio:
                base = self.ratio * h - w
                white_one = np.zeros([h, int(base * x), 3], dtype=np.uint8)
                white_one.fill(self.color)
                white_two = np.zeros([h, int(base * (1 - x)), 3], dtype=np.uint8)
                white_two.fill(self.color)
                img = cv2.hconcat([white_one, img, white_two])
            elif (w / h) > self.ratio:
                base = (w - self.ratio * h) // self.ratio
                white_one = np.zeros([int(base * x), w, 3], dtype=np.uint8)
                white_one.fill(self.color)
                white_two = np.zeros([int(base * (1 - x)), w, 3], dtype=np.uint8)
                white_two.fill(self.color)
                img = cv2.vconcat([white_one, img, white_two])
        img = cv2.resize(img, (self.width, self.height))

        return img


transforms = A.Compose([

    A.Resize(384, 512, p=1.0),

    A.OneOf([
        A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.1, p=0.75),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.75),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    ], p=0.75),

    A.OneOf([
        A.Blur(blur_limit=2.5, p=0.75),
        A.MotionBlur(blur_limit=(3, 5), p=0.75),
    ], p=0.75),

    A.OneOf([
        A.Affine(shear=21, mode=cv2.BORDER_REPLICATE, p=0.75),
        A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
            rotate_limit=16, border_mode=cv2.BORDER_REPLICATE, p=0.75),
            A.GridDistortion(distort_limit=0.1, border_mode=cv2.BORDER_REPLICATE, p=0.75),
        ], p=0.75)
    ], p=0.75),

    A.OneOf([
        A.Cutout(num_holes=12, max_h_size=16, max_w_size=16, p=0.75),
        A.Cutout(num_holes=6, max_h_size=32, max_w_size=32, p=0.75),
    ], p=0.75),
    
    A.CLAHE(clip_limit=2.5, p=0.75),
    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.75),
    A.Normalize(p=1.0),
])

path = '/home/toefl/K/dolphin/datasets/backfins/train_images'
for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))

    plt.imshow(img)
    plt.show()

    # resize = SmartResize(512, 384, stretch=(1.0, 1.0), fillcolor=255)
    # img = resize(img)
    img = transforms(image=img)["image"]
    # img = color.rgb2gray(img)

    plt.imshow(img)
    plt.show()