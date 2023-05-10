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
from img_utils import load_img_and_convert_to_three_channels, save_image_with_masks, get_colour_freqs
import statistics
# from generate_imgs import ball_area

ball_rad = 15 # default radius, be wary if this changes
ball_area = math.pi * (ball_rad ** 2)

def print_mask_info(masks_pkl_path, image):
    print("INSPECTING THE MASKS FOUND IN " + masks_pkl_path)
    with open(masks_pkl_path, "rb") as f:
        masks = pickle.load(f)
    masks = add_masks_colours(masks, image)
    print("No of masks: {}".format(len(masks)))
    mask_areas = list(map(lambda mask: mask['area'], masks))
    print("Mask areas: " + str(mask_areas))
    colours = list(map(lambda mask: mask['colour'], masks))
    print("Mask colours: " + str(colours))


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

def add_masks_colours(masks, image):
    plt.figure(figsize=(20, 20))
    for mask in masks:
        [x, y, width, height] = mask['bbox']
        x_centre = x + math.floor(width / 2)
        y_centre = y + math.floor(height / 2)
        rgb_colour = image[y_centre, x_centre, :]
        mask['colour'] = webcolors.rgb_to_name(rgb_colour)
    return masks

# So far, this method gets the events by looking at the area of every mask and seeing if there is a mask that is smaller, potentially implying that this corresponds to a ball that has something intersecting with it
def get_events_from_masks_in_state(event_vocab, masks, image):
    events = set()
    print("The number of masks extracted is {}".format(len(masks)))
    mask_areas = np.array(list(map(lambda mask: mask['area'], masks)))
    abnormal_mask_indices, common_mask_area = extract_abnormal_masks(mask_areas)
    if abnormal_mask_indices == {}:
        return events
    for ix in abnormal_mask_indices:
        mask_pixels = masks[ix]['segmentation']
        colour_distribution = get_colour_freqs(mask_pixels, image)
        if masks[ix]['area'] < common_mask_area:
            print("A smaller mask has been found")
            event = list(colour_distribution.keys())[0]
            if (event in event_vocab):
                # This is hardcoded in the frozen world because the agent (white) is the only ball that is moving. When we remove the frozen ball assumption, I hope that we don't run into the problem of having a mask covering most of one ball but not the other
                events.add((event, 'white'))
        else:
            print("A bigger mask has been found")
            largest_colour_presences = sorted(colour_distribution, reverse=True)[:2]
            if largest_colour_presences[0] in event_vocab and largest_colour_presences[1] in event_vocab:
                events.add((largest_colour_presences[0], largest_colour_presences[1]))
    return events

# Right now, I get the most common mask area because most of the masks will be just one ball. When multiple balls are moving, this may need to change by injecting the knowledge of what we know the ball area to be (which you would have to get when you get the event vocab because the pixel area is not perfect)
def extract_abnormal_masks(mask_areas):
    common_mask_area = statistics.mode(mask_areas)
    ixs = []
    for i in range(len(mask_areas)):
        mask_area = mask_areas[i]
        # Allow for some leeway in the mask area because of potentially poor image resolution
        if mask_area < common_mask_area - 5 or mask_area > common_mask_area + 5:
            ixs.append(i)
    return ixs, common_mask_area

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
def run_segmentation_on_single_image(img_path, sam_checkpoint, model_type, masks_pkl_path=None, unfiltered_masks_pkl_path=None):
    print("Loading image of the environment")
    image = load_img_and_convert_to_three_channels(img_path)
    print("Generating masks for the image using Segment Anything (" + model_type + ")")
    masks = generate_masks(image, sam_checkpoint, model_type)
    if unfiltered_masks_pkl_path is not None:
        with open(unfiltered_masks_pkl_path, "wb") as f:
            pickle.dump(masks, f)
    print("Filtering the masks generated")
    filtered_masks = filter_masks_based_on_size(masks, lower_uncertainty=100, upper_uncertainty=200)
    if unfiltered_masks_pkl_path is not None:
        with open(unfiltered_masks_pkl_path, "wb") as f:
            pickle.dump(masks, f)
    if masks_pkl_path is not None:
        with open(masks_pkl_path, "wb") as f:
            pickle.dump(filtered_masks, f)
    return image, masks, filtered_masks

def segment_and_save_image_with_masks(img_path, sam_checkpoint, model_type, masks_pkl_path, img_with_filtered_masks_path, unfiltered_masks_pkl_path, img_with_unfiltered_masks_path):
    image, unfiltered_masks, filtered_masks = run_segmentation_on_single_image(img_path, sam_checkpoint, model_type, masks_pkl_path, unfiltered_masks_pkl_path)
    save_image_with_masks(filtered_masks, image, img_with_filtered_masks_path)
    save_image_with_masks(unfiltered_masks, image, img_with_unfiltered_masks_path)

if __name__ == "__main__":
    dir_path = "single_img_experimentation/eg_ww_img/"
    orig_img_name = "env_step0.png"
    eg_img_path = dir_path + orig_img_name
    sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    masks_pkl_path = dir_path + "env_step_masks_filtered_on_size_" + model_type + ".pkl"
    unfiltered_masks_pkl_path = dir_path + "env_step_unfiltered_masks_" + model_type + ".pkl"
    img_with_filtered_masks_path = dir_path + "env_step_filtered_masks_" + model_type + ".png"
    img_with_unfiltered_masks_path = dir_path + "env_step_unfiltered_masks_" + model_type + ".png"

    segment_and_save_image_with_masks(eg_img_path, sam_checkpoint, model_type, masks_pkl_path, img_with_filtered_masks_path, unfiltered_masks_pkl_path, img_with_unfiltered_masks_path)

    # image = load_img_and_convert_to_three_channels(eg_img_path)
    # print_mask_info(unfiltered_masks_pkl_path, image)
    # hardcoded_event_vocab = set()
    # hardcoded_event_vocab.add('red')
    # hardcoded_event_vocab.add('lime')
    # hardcoded_event_vocab.add('cyan')
    # hardcoded_event_vocab.add('magenta')
    # hardcoded_event_vocab.add('yellow')
    # hardcoded_event_vocab.add('blue')
    # hardcoded_event_vocab.add('white')

    # with open(masks_pkl_path, "rb") as f:
    #     masks = pickle.load(f)
    # image = load_img_and_convert_to_three_channels(eg_img_path)
    # events = get_events_from_masks_in_state(hardcoded_event_vocab, masks, image)
    # print(events)