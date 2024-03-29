import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import webcolors

# Loads an image and converts it to a 3 channel RBG image
def load_img_and_convert_to_three_channels(path_to_img):
    img = np.asarray(Image.open(path_to_img))
    three_channel_img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    return three_channel_img

# Saves a given image to the specified directory
def save_image(img_numpy, directory):
    image = Image.fromarray(img_numpy, "RGB")
    image.save(directory)

# Converts a colour in RGB format to its hex value
def convert_rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

# Display image with axis
def save_image_with_axis(img, path):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('on')
    plt.savefig(path)

# Display image
def show_image(img, masks=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    if masks is not None:
        show_anns(masks)
    plt.axis('off')
    plt.show()

# Show image with masks
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        mask_on_image = np.dstack((img, m * 0.35))
        ax.imshow(mask_on_image)

# Save a snapshot of the WaterWorld environment with the masks that have automatically generated with Segment Anything
def save_image_with_masks(masks, image, path, event=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    for mask in masks:
        [x, y, width, height] = mask['bbox']
        mask_boundary = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax = plt.gca()
        ax.add_patch(mask_boundary)
    if event is not None:
        plt.title(str(event), fontsize=30)
    plt.savefig(path)
    plt.close()

# Construct and return a dictionary that visualises the colour distribution in a mask
def get_colour_freqs(mask, image):
    colour_freq = {}
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j]:
                rgb = image[i][j]
                colour = webcolors.rgb_to_name(rgb)
                colour_freq[colour] = colour_freq.get(hex, 0) + 1
    return colour_freq