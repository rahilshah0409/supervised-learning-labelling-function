import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image

def load_img_and_convert_to_three_channels(path_to_img):
    img = np.asarray(Image.open(path_to_img))
    three_channel_img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return three_channel_img

def save_image(img_numpy, directory):
    image = Image.fromarray(img_numpy, "RGB")
    image.save(directory)