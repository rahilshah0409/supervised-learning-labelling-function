import string
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
sys.path.insert(1, "../")
from image_segmentation.img_utils import load_img_and_convert_to_three_channels, save_image_with_masks, get_colour_freqs
import statistics

ball_rad = 15 # default radius, be wary if this changes
ball_area = math.pi * (ball_rad ** 2)
    
# Prints the information about the masks
def print_mask_info(masks_pkl_path, image):
    print("INSPECTING THE MASKS FOUND IN " + masks_pkl_path)
    with open(masks_pkl_path, "rb") as f:
        masks = pickle.load(f)
    print("No of masks: {}".format(len(masks)))
    mask_areas = list(map(lambda mask: mask['area'], masks))
    print("Mask areas: " + str(mask_areas))
    colours = list(map(lambda mask: mask['colour'], masks))
    print("Mask colours: " + str(colours))

# Creates event configuration from objects detected in a state where no event is assumed to be observed
def create_event_vocab(masks, image):
    centres = []
    vocab = set()
    freq_of_colours = {}
    ball_centres = {}
    common_mask_area = 0
    for mask in masks:
        [x, y, width, height] = mask['bbox']
        common_mask_area = mask['area']
        x_centre = x + math.floor(width / 2)
        y_centre = y + math.floor(height / 2)
        centres.append((x_centre, y_centre))
        rgb_colour = image[y_centre, x_centre, :]
        colour_name = webcolors.rgb_to_name(rgb_colour)
        vocab.add(colour_name)
        if colour_name in ball_centres:
            ball_centres[colour_name].append((x_centre, y_centre))
        else:
            ball_centres[colour_name] = [(x_centre, y_centre)]
        if colour_name in freq_of_colours:
            freq_of_colours[colour_name] += 1
        else:
            freq_of_colours[colour_name] = 1
    expected_radius = math.sqrt(common_mask_area / math.pi)
    return vocab, freq_of_colours, ball_centres, common_mask_area, expected_radius

# Collision detection algorithm- given the objects identified for a given state, this algorithm extracts the events that have been observed at that state (using previously extracted knowledge)
def get_events_from_masks_in_state(event_vocab, masks, image, expected_freq_of_objs, expected_no_of_objs, common_obj_size, expected_ball_centres, expected_radius):
    no_of_masks_missing = expected_no_of_objs - len(masks)
    freq_of_objs = {}
    events = set()
    ball_centres = []
    for mask in masks:
        mask_area = mask['area']
        mask_pixels = mask['segmentation']
        colour_distribution = get_colour_freqs(mask_pixels, image)
        [x, y, width, height] = mask['bbox']
        # Give some leeway due to potentially poor image resolution
        if mask_area > common_obj_size + 5:
            print("A bigger mask has been observed")
            # We only consider two because we assume, in the frozen setting, that a big mask can only be made up of two masks. This can be made to be more general in the dynamic setting
            largest_colour_presences = sorted(colour_distribution, reverse=True)[:2]
            events = add_pair_to_events(events, event_vocab, (largest_colour_presences[0], largest_colour_presences[1]))
            freq_of_objs[largest_colour_presences[0]] = 1 if largest_colour_presences[0] not in freq_of_objs else freq_of_objs[largest_colour_presences[0]] + 1
            freq_of_objs[largest_colour_presences[1]] = 1 if largest_colour_presences[1] not in freq_of_objs else freq_of_objs[largest_colour_presences[1]] + 1
        else:
            colour = list(colour_distribution.keys())[0]
            freq_of_objs[colour] = 1 if colour not in freq_of_objs else freq_of_objs[colour] + 1
            x_centre = x + math.floor(width / 2)
            y_centre = y + math.floor(height / 2)
            ball_centres.append((colour, (x_centre, y_centre)))
    for i in range(len(ball_centres)):
        for j in range(i + 1, len(ball_centres)):
            (c1, (x1, y1)) = ball_centres[i]
            (c2, (x2, y2)) = ball_centres[j]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance <= expected_radius * 2:
                events = add_pair_to_events(events, event_vocab, (c1, c2))
    # Get the events that can be observed from no longer seeing a mask
    missing_colours = set(expected_freq_of_objs.keys()).difference(set(freq_of_objs.keys()))
    for missing_colour in missing_colours:
        expected_freq_of_colour = expected_freq_of_objs[missing_colour]
        for i in range(expected_freq_of_colour):
            events = add_pair_to_events(events, event_vocab, ('black', missing_colour))
    for colour in freq_of_objs.keys():
        freq = freq_of_objs[colour]
        # There will only be a difference of one in the WaterWorld setting that we are working in. This can simply be adapted if this wasn't the case
        if colour in expected_freq_of_objs and expected_freq_of_objs[colour] > freq:
            # In the frozen setting, we know that the black agent ball is the only ball capable of making a mask disappear, so we know the missing colour must have overlapped with black
            events = add_pair_to_events(events, event_vocab, (colour, 'black'))
    return events

# Helper method that adds the event pair into a set of events that is then returned
def add_pair_to_events(events, event_vocab, e_pair):
    pair = e_pair
    if e_pair[0] > e_pair[1]:
        pair = (e_pair[1], e_pair[0])
    if pair[0] in event_vocab and pair[1] in event_vocab:
        events.add(pair)
    return events

# Gets the indices of abnormally sized masks
def extract_abnormal_masks(mask_areas):
    common_mask_area = statistics.mode(mask_areas)
    ixs = []
    for i in range(len(mask_areas)):
        mask_area = mask_areas[i]
        # Allow for some leeway in the mask area because of potentially poor image resolution
        if mask_area < common_mask_area - 5 or mask_area > common_mask_area + 5:
            ixs.append(i)
    return ixs, common_mask_area

# Performs image segmentation on a single image
def generate_masks(image, sam_checkpoint, model_type):
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks

# Helper method that filters out abnormally sized masks produced by Segment Anything
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
    if masks_pkl_path is not None:
        with open(masks_pkl_path, "wb") as f:
            pickle.dump(filtered_masks, f)
    return image, masks, filtered_masks

# Performs object detection with Segment Anything and saves the image with the masks identified
def segment_and_save_image_with_masks(img_path, sam_checkpoint, model_type, filtered_masks_pkl_path, img_with_filtered_masks_path, unfiltered_masks_pkl_path=None, img_with_unfiltered_masks_path=None):
    image, _, filtered_masks = run_segmentation_on_single_image(img_path, sam_checkpoint, model_type, filtered_masks_pkl_path, unfiltered_masks_pkl_path)
    save_image_with_masks(filtered_masks, image, img_with_filtered_masks_path)

# Method that checks how accurately all individual balls are identified in a dataset of images
def check_accuracy_of_ball_detection(data_dir, expected_no_of_balls, expected_ball_area):
    with open(data_dir + "traces_data.pkl", "rb") as f:
        trace_data = pickle.load(f)
    num_eps = len(trace_data)
    inaccurate_states = 0
    accurate_states = 0
    for ep in range(num_eps):
        sub_dir = data_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]['length']
        for step in range(ep_len):
            masks_i_loc = sub_dir + "masks" + str(step) + ".pkl"
            with open(masks_i_loc, "rb") as f:
                masks = pickle.load(f)
            if len(masks) != expected_no_of_balls:
                inaccurate_states += 1
                break
            abnormally_sized_mask_found = False
            for mask in masks:
                if mask['area'] < expected_ball_area - 5 or mask['area'] > expected_ball_area + 5:
                    inaccurate_states += 1
                    abnormally_sized_mask_found = True
                    break
            if not abnormally_sized_mask_found:
                accurate_states += 1
    return accurate_states, inaccurate_states

if __name__ == "__main__":
    dir_path = "single_img_experimentation/touching_balls/"
    orig_img_name = "touching_balls"
    eg_img_path = dir_path + orig_img_name + ".png"
    sam_checkpoint = "/vol/bitbucket/ras19/fyp/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    masks_pkl_path = dir_path + orig_img_name + "_masks.pkl"
    img_with_filtered_masks_path = dir_path + orig_img_name + "_masks.png"

    segment_and_save_image_with_masks(eg_img_path, sam_checkpoint, model_type, masks_pkl_path, img_with_filtered_masks_path)

    image = load_img_and_convert_to_three_channels(eg_img_path)

    with open(masks_pkl_path, "rb") as f:
        masks = pickle.load(f)
    image = load_img_and_convert_to_three_channels(eg_img_path)
    # events = get_events_from_masks_in_state(hardcoded_event_vocab, masks, image)