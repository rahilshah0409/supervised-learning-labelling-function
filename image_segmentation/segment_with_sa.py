from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import math
import webcolors
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
# The two paths below are hardcoded from my local machine and hence will be useless when working on lab machines
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/image-segmentation-experimentation/image_segmentation")
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/image-segmentation-experimentation/image_generation")
sys.path.insert(1, "..")
from img_utils import load_img_and_convert_to_three_channels, get_colour_freqs
# from generate_imgs import ball_area

ball_rad = 15 # default radius, be wary if this changes
ball_area = math.pi * (ball_rad ** 2)

def inspect_masks(masks, image):
    print("The number of masks extracted is {}".format(len(masks)))
    mask_areas = list(map(lambda mask: mask['area'], masks))
    print(mask_areas)
    masks_pixels = list(map(lambda mask: mask['segmentation'], masks))
    colour_freqs = list(map(lambda mpixels: get_colour_freqs(image, mpixels), masks_pixels))
    print(colour_freqs)


# Either can look at the individual pixels or look at the centre of the box that surrounds the box. Prefer the latter. Assuming x increases as you go to the right and y increases as you go down. Pretty sure the flooring doesn't matter in this function, it is just so I can get whole numbers that I can then use as co-ords
def create_event_vocab(masks, image):
    plt.figure(figsize=(20, 20))
    centres = []
    vocab = set()
    for mask in masks:
        [x, y, width, height] = mask['bbox']
        x_centre = x + math.floor(width / 2)
        y_centre = y + math.floor(height / 2)
        centres.append((x_centre, y_centre))
        rgb_colour = image[y_centre, x_centre, :]
        mask['colour'] = rgb_colour
        vocab.add(webcolors.rgb_to_name(rgb_colour))
    return vocab

# So far, this method gets the events by looking at the area of every mask and seeing if there is a mask that is smaller, potentially implying that this corresponds to a ball that has something intersecting with it
def get_events_from_masks_in_state(event_vocab, masks, image):
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

# Produces masks for a single image, before filtering the masks to extract the relevant ones and saving these masks in a pickle object for later use
def run_segmentation_on_single_image(img_path, sam_checkpoint, model_type, masks_pkl_dir, save_unfiltered_masks=False):
    print("Loading image of the environment")
    image = load_img_and_convert_to_three_channels(img_path)
    print("Generating masks for the image using Segment Anything (" + model_type + ")")
    masks = generate_masks(image, sam_checkpoint, model_type)
    if save_unfiltered_masks:
        with open(masks_pkl_dir + "unfiltered_masks_" + model_type + ".pkl", "wb") as f:
            pickle.dump(masks, f)
    print("Filtering the masks generated")
    filtered_masks = filter_masks_based_on_size(masks, lower_uncertainty=100, upper_uncertainty=200)
    with open(masks_pkl_dir + "masks_filtered_on_size_" + model_type + ".pkl", "wb") as f:
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
    # inspect_masks(masks, image)
