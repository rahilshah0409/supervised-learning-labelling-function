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
# from generate_imgs import ball_area

ball_rad = 15 # default radius, be wary if this changes
ball_area = math.pi * (ball_rad ** 2)

class EventLabel(object):
    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2

    def __eq__(self, other_object):
        return self.__class__ == other_object.__class__ and (other_object == (self.object1, self.object2) or other_object == (self.object2, self.object1))
    
    def __ne__(self, other_object):
        return not self == other_object
    
    def __hash__(self):
        return hash(self.get_pair())
    
    def __str__(self):
        return "(" + self.object1 + "," + self.object2 + ")"
    
    def __repr__(self):
        return "(" + self.object1 + "," + self.object2 + ")"
    
    def get_pair(self):
        return (self.object1, self.object2)
    

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
    centres = []
    vocab = set()
    freq_of_colours = {}
    common_mask_area = 0
    # This is hardcoded to overcome an obstacle where the agent doesn't get picked up. This hard coding should be removed with a better solution- either we make the ball opaque or figure out a way to get the agent in somehow
    for mask in masks:
        # Looking at just the centre is probably flawed, not in the waterworld where the objects are clean but in other environments where the objects are not so exact
        [x, y, width, height] = mask['bbox']
        # Making an assumption that all the masks will be of the same size, and this is the size that the masks should be
        common_mask_area = mask['area']
        x_centre = x + math.floor(width / 2)
        y_centre = y + math.floor(height / 2)
        centres.append((x_centre, y_centre))
        rgb_colour = image[y_centre, x_centre, :]
        colour_name = webcolors.rgb_to_name(rgb_colour)
        vocab.add(colour_name)
        if colour_name in freq_of_colours:
            freq_of_colours[colour_name] += 1
        else:
            freq_of_colours[colour_name] = 1
    return vocab, freq_of_colours, common_mask_area

def add_masks_colours(masks, image):
    plt.figure(figsize=(20, 20))
    for mask in masks:
        # [x, y, width, height] = mask['bbox']
        # x_centre = x + math.floor(width / 2)
        # y_centre = y + math.floor(height / 2)
        # rgb_colour = image[y_centre, x_centre, :]
        colour_freqs = get_colour_freqs(mask['segmentation'], image)
        all_colours_in_mask = list(colour_freqs.keys())
        mask['colour'] = all_colours_in_mask
        # mask['colour'] = webcolors.rgb_to_name(rgb_colour)
    return masks

# So far, this method gets the events by looking at the area of every mask and seeing if there is a mask that is smaller, potentially implying that this corresponds to a ball that has something intersecting with it
# Method extracts events being observed in a state by looking at three edge cases:
# 1. There is an abnormally small mask
# 2. There is an abnormally large mask
# 3. There are less masks than what is expected and an event was observed previously
# past_events is a set to not have any constraints on how long the agent ball can overlap with any coloured ball
# expected_no_of_objects is obtained from the initial state
def _get_events_from_masks_in_state(event_vocab, masks, image, expected_freq_of_objs, expected_no_of_objs):
    events = set()
    less_masks = False
    print("The number of masks extracted is {}".format(len(masks)))
    if expected_no_of_objects > len(masks):
        no_of_missing_objects = expected_no_of_objects - len(masks)
        # It is quite likely that the agent has started to overlap with a ball of colour X more, so ball of colour X is no longer being picked up as a mask
        # Likelihood quite high because there is little noise in terms of relevant masks that are picked up by SA (especially after filtering)
        print("We are expecting {} objects but we have only identified {}".format(expected_no_of_objects, len(masks)))
        less_masks = True
        #for i in range(no_of_missing_objects):
        #    if i < len(past_events):
        #        events = add_pair_to_events(events, past_events[i])
        for event_pair in past_events:
            events = add_pair_to_events(events, event_pair)
    elif events_in_prev_state:
        for event_pair in events_in_prev_state:
            if event_pair in past_events:
                past_events.remove(event_pair)
    elif less_masks_in_prev_state:
        # A small mask was detected at the beginning of the agent ball's overlapping with another ball but not at the end
        past_events = set()
    mask_areas = np.array(list(map(lambda mask: mask['area'], masks)))
    abnormal_mask_indices, common_mask_area = extract_abnormal_masks(mask_areas)
    if abnormal_mask_indices == {}:
        return events
    for ix in abnormal_mask_indices:
        mask_pixels = masks[ix]['segmentation']
        colour_distribution = get_colour_freqs(mask_pixels, image)
        if masks[ix]['area'] < common_mask_area:
            print("A smaller mask has been found") # Two balls are overlapping with each other slightly
            event = list(colour_distribution.keys())[0]
            if event in event_vocab:
                # TODO: Generalise this so that any two balls can overlap with each other and for that to be detected in this case
                event_to_add = (event, 'black') if event <= 'black' else ('black', event)
                events = add_pair_to_events(events, event_to_add)
                if event_to_add not in events_in_prev_state:
                    if less_masks_in_prev_state:
                        if event_to_add in past_events:
                            # If there were less masks in the previous state then you need to remove the event. If the event wasn't added in the first place, then that's fine
                            past_events.remove(event_to_add)
                    else:
                        # The agent ball is only starting to overlap with the coloured ball
                        # past_events.insert(0, event_to_add)
                        past_events = add_pair_to_events(past_events, event_to_add)
                else:
                    past_events = add_pair_to_events(past_events, event_to_add)
        else:
            print("A bigger mask has been found") # Probably found because two balls are really close to each other and overlapping a lot
            largest_colour_presences = sorted(colour_distribution, reverse=True)[:2]
            if largest_colour_presences[0] in event_vocab and largest_colour_presences[1] in event_vocab:
                events = add_pair_to_events(events, (largest_colour_presences[0], largest_colour_presences[1]))
    return events, past_events, less_masks

def get_events_from_masks_in_state(event_vocab, masks, image, expected_freq_of_objs, expected_no_of_objs, common_obj_size):
    no_of_masks_missing = expected_no_of_objs - len(masks)
    freq_of_objs = {}
    events = set()
    for mask in masks:
        mask_area = mask['area']
        mask_pixels = mask['segmentation']
        colour_distribution = get_colour_freqs(mask_pixels, image)
        # Give some leeway due to potentially poor image resolution
        if mask_area > common_obj_size + 5:
            print("A bigger mask has been observed")
            # We only consider two because we assume, in the frozen setting, that a big mask can only be made up of two masks. This can be made to be more general in the dynamic setting
            largest_colour_presences = sorted(colour_distribution, reverse=True)[:2]
            if largest_colour_presences[0] in event_vocab and largest_colour_presences[1] in event_vocab:
                events = add_pair_to_events(events, (largest_colour_presences[0], largest_colour_presences[1]))
                freq_of_objs[largest_colour_presences[0]] = 1 if largest_colour_presences[0] not in freq_of_objs else freq_of_objs[largest_colour_presences[0]] + 1
                freq_of_objs[largest_colour_presences[1]] = 1 if largest_colour_presences[1] not in freq_of_objs else freq_of_objs[largest_colour_presences[1]] + 1
        elif mask_area < common_obj_size - 5:
            print("A smaller mask has been observed")
            event = list(colour_distribution.keys())[0]
            if event in event_vocab:
                # TODO: Generalise this so that any two balls can overlap with each other and for that to be detected in this case
                event_to_add = (event, 'black') if event <= 'black' else ('black', event)
                events = add_pair_to_events(events, event_to_add)
                freq_of_objs[event] = 1 if event not in freq_of_objs else freq_of_objs[event] + 1
        else:
            colour = list(colour_distribution.keys())[0]
            freq_of_objs[colour] = 1 if colour not in freq_of_objs else freq_of_objs[colour] + 1
    missing_colours = set(expected_freq_of_objs.keys()).difference(set(freq_of_objs.keys()))
    for missing_colour in missing_colours:
        expected_freq_of_colour = expected_freq_of_objs[missing_colour]
        for i in range(expected_freq_of_colour):
            events = add_pair_to_events(events, ('black', missing_colour))
    for colour in freq_of_objs.keys():
        freq = freq_of_objs[colour]
        # There will only be a difference of one in the WaterWorld setting that we are working in. This can simply be adapted if this wasn't the case
        if colour in expected_freq_of_objs and expected_freq_of_objs[colour] > freq:
            # In the frozen setting, we know that the black agent ball is the only ball capable of making a mask disappear, so we know the missing colour must have overlapped with black
            # In the dynamic setting, how would you know what two balls overlap with each other (look into the past or future?)
            events = add_pair_to_events(events, (colour, 'black'))
    return events

def add_pair_to_events(events, e_pair):
    pair = e_pair
    if e_pair[0] > e_pair[1]:
        pair = (e_pair[1], e_pair[0])
    events.add(pair)
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
    masks_pkl_path = dir_path + "masks_filtered_on_size_" + model_type + ".pkl"
    unfiltered_masks_pkl_path = dir_path + "unfiltered_masks_" + model_type + ".pkl"
    img_with_filtered_masks_path = dir_path + "filtered_masks_" + model_type + ".png"
    img_with_unfiltered_masks_path = dir_path + "unfiltered_masks_" + model_type + ".png"

    # segment_and_save_image_with_masks(eg_img_path, sam_checkpoint, model_type, masks_pkl_path, img_with_filtered_masks_path, unfiltered_masks_pkl_path, img_with_unfiltered_masks_path)

    image = load_img_and_convert_to_three_channels(eg_img_path)
    print_mask_info(unfiltered_masks_pkl_path, image)
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