import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.insert(1, "..")
import img_utils as utils

def plot_threshold_vs_grey_vs_original(thresh, gray, original):
    fig, (axs1, axs2, axs3) = plt.subplots(1, 3)
    fig.suptitle("Comparing threshold plot with the original image")
    axs1.imshow(thresh, 'gray', vmin=0, vmax=255)
    axs1.set(title="Threshold plot", xticks=[], yticks=[])
    axs2.imshow(gray, vmin=0, vmax=255)
    axs2.set(title="Grayed out image", xticks=[], yticks=[])
    axs3.imshow(original, vmin=0, vmax=255)
    axs3.set(title="Original image", xticks=[], yticks=[])
    plt.show()

def plot_thresholds_against_original(thresh_arr, subplot_titles, original):
    fig, axs = plt.subplots(len(thresh_arr), 2)
    for i in range(len(thresh_arr)):
        axs[i, 0].imshow(thresh_arr[i], 'gray', vmin=0, vmax=255)
        axs[i, 0].set(title=subplot_titles[i], xticks=[], yticks=[])
        axs[i, 1].imshow(original, vmin=0, vmax=255)
        axs[i, 1].set(title="Original image", xticks=[], yticks=[])
    plt.show()

def approximate_estimate(path_to_img, otsu_binarisation=True):
    img = cv.imread(path_to_img)
    assert img is not None, "File could not be read, check with os.path.exists()"
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # threshold = cv.THRESH_BINARY_INV+cv.THRESH_OTSU if otsu_binarisation else cv.THRESH_BINARY_INV
    threshold = cv.THRESH_TOZERO
    ret, thresh = cv.threshold(gray, 0, 255, threshold)
    return ret, thresh, gray

def extract_sure_fg_and_bg(thresh):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    return sure_fg, sure_bg, unknown

def create_markers(sure_fg, unknown):
    ret, markers =  cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    return markers

if __name__ == "__main__":
    eg_image_path = "../waterworld_imgs/example.png"
    three_channel_img = utils.load_img_and_convert_to_three_channels(eg_image_path)
    ret, thresh, gray = approximate_estimate(eg_image_path)
    plot_threshold_vs_grey_vs_original(thresh, gray, three_channel_img)
    sure_fg, sure_bg, unknown = extract_sure_fg_and_bg(thresh)
    plot_thresholds_against_original([sure_fg, sure_bg], ["Sure foreground", "Sure background"], three_channel_img)
    markers = create_markers(sure_fg, unknown)
    markers = cv.watershed(three_channel_img, markers)
    three_channel_img[markers == -1] = [255, 0, 0]
    plt.imshow(three_channel_img, vmin=0, vmax=255)
    plt.show()
