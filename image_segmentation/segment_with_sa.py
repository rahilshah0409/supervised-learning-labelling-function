from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import math
import webcolors
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import sys
# The two paths below are hardcoded from my local machine and hence will be useless when working on lab machines
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/image-segmentation-experimentation/image_segmentation")
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/image-segmentation-experimentation/image_generation")
sys.path.insert(1, "..")
from img_utils import load_img_and_convert_to_three_channels
# from generate_imgs import ball_area

ball_rad = 15 # default radius, be wary if this changes
ball_area = math.pi * (ball_rad ** 2)

# def save_image_with_masks(img_original, masks):
#     original_image = Image.fromarray(img_original)
#     if len(masks) == 0:
#         return
#     sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         img = np.ones((m.shape[0], m.shape[1], 3))
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:, :, i] = color_mask[i]
#         img_with_one_mask = np.dstack((img, m * 0.35))
#         print("Dimensions of image overlayed with a mask")
#         print(img_with_one_mask.shape[0])
#         print(img_with_one_mask.shape[1])
#         print(img_with_one_mask.shape[2])
#         # original_image.paste(img_with_one_mask)
#     # original_image.save("../saved_images/segment_anything.jpg")

def save_image_with_masks(masks, image, path, event=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('on')
    for mask in masks:
        [x, y, width, height] = mask['bbox']
        mask_boundary = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax = plt.gca()
        ax.add_patch(mask_boundary)
    if event is not None:
        plt.title(str(event))
    plt.savefig(path)
    plt.close()

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


def show_image(img, masks=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    if masks is not None:
        show_anns(masks)
    plt.axis('off')
    plt.show()


def save_image_with_axis(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('on')
    plt.savefig("../eg_ww_img/example_with_axis.png")

def generate_and_filter_masks(image, sam_checkpoint, model_type):
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masks = filter_masks_based_on_size(masks, lower_uncertainty=100, upper_uncertainty=200)
    return masks

def generate_and_save_masks(image, sam_checkpoint, model_type, pkl_path):
    masks = generate_and_filter_masks(image, sam_checkpoint, model_type)
    masks, centres = find_mask_colours(masks, image)
    with open(pkl_path, "wb") as f:
        pickle.dump(masks, f)
    return masks, centres

def convert_rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def get_colour_freqs(mask):
    colour_freq = {}
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j]:
                bgr = image[i][j]
                hex = convert_rgb_to_hex(bgr)
                colour_freq[hex] = colour_freq.get(hex, 0) + 1
    return colour_freq

def inspect_masks(masks):
    print("The number of masks extracted is {}".format(len(masks)))
    mask_areas = list(map(lambda mask: mask['area'], masks))
    print(mask_areas)
    masks_pixels = list(map(lambda mask: mask['segmentation'], masks))
    colour_freqs = list(map(get_colour_freqs, masks_pixels))
    print(colour_freqs)


# Either can look at the individual pixels or look at the centre of the box that surrounds the box. Prefer the latter. Assuming x increases as you go to the right and y increases as you go down. Pretty sure the flooring doesn't matter in this function, it is just so I can get whole numbers that I can then use as co-ords
def find_mask_colours(masks, image):
    plt.figure(figsize=(20, 20))
    centres = []
    colours = set()
    for mask in masks:
        [x, y, width, height] = mask['bbox']
        x_centre = x + math.floor(width / 2)
        y_centre = y + math.floor(height / 2)
        centres.append((x_centre, y_centre))
        rgb_colour = image[y_centre, x_centre, :]
        mask['colour'] = rgb_colour
        colours.add(webcolors.rgb_to_name(rgb_colour))
    return masks, centres, colours

# So far, this method gets the events by looking at the area of every mask and seeing if there is a mask that is smaller, potentially implying that this corresponds to a ball that has something intersecting with it
def get_event_occured(event_vocab, masks, image):
    events = set()
    print("The number of masks extracted is {}".format(len(masks)))
    mask_areas = np.array(list(map(lambda mask: mask['area'], masks)))
    if mask_areas.min() < mask_areas.max():
        small_mask_ix = mask_areas.argmin()
        mask = masks[small_mask_ix]
        [x, y, width, height] = mask['bbox']
        x_centre = x + math.floor(width / 2)
        y_centre = y + math.floor(height / 2)
        rgb_colour = image[y_centre, x_centre, :]
        colour_name = webcolors.rgb_to_name(rgb_colour)
        if colour_name in event_vocab:
            events.add(colour_name)
    return events

def generate_masks(image, sam_checkpoint, model_type):
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks

def filter_masks_based_on_size(masks, lower_uncertainty, upper_uncertainty):
    filtered_masks = list(filter(lambda m: m['area'] >= ball_area - lower_uncertainty and m['area'] <= ball_area + upper_uncertainty, masks))
    return filtered_masks

def run_segmentation_on_single_image(img_path, sam_checkpoint, model_type, masks_pkl_dir, save_unfiltered_masks=False):
    print("Loading image of the environment")
    image = load_img_and_convert_to_three_channels(img_path)
    print("Generating masks for the image using Segment Anything (" + model_type + ")")
    masks = generate_masks(image, sam_checkpoint, model_type)
    if save_unfiltered_masks:
        with open(masks_pkl_dir + "unfiltered_masks.pkl", "wb") as f:
            pickle.dump(masks, f)
    print("Filtering the masks generated")
    filtered_masks = filter_masks_based_on_size(masks, lower_uncertainty=100, upper_uncertainty=200)
    with open(masks_pkl_dir + "masks_filtered_on_size.pkl", "wb") as f:
        pickle.dump(filtered_masks, f)

if __name__ == "__main__":
    dir_path = "../single_img_experimentation/colliding_ww_img/"
    orig_img_name = "colliding_example.png"
    masks_pkl_filename = "masks_small_filter.pkl"
    img_with_masks_filename = "example_with_masks_small_filter.png"
    eg_img_path = dir_path + orig_img_name
    pkl_path = dir_path + masks_pkl_filename
    sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    run_segmentation_on_single_image(eg_img_path, sam_checkpoint, model_type, dir_path, save_unfiltered_masks=True)

    # with open(pkl_path, "rb") as f:
    #     masks = pickle.load(f)
    # show_image(image, masks)
    # save_image_with_masks(masks, image, dir_path + img_with_masks_filename)
    # inspect_masks(masks)
