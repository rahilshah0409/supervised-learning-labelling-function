import pickle
import sys
sys.path.insert(1, "segment_anything/")
from segment_with_sa import create_event_vocab, get_events_from_masks_in_state, save_image_with_masks, run_segmentation_on_single_image, add_masks_colours
from img_utils import load_img_and_convert_to_three_channels

def generate_and_save_masks_for_eps(trace_data, trace_dir, sam_checkpoint, model_type, img_base_filename):
    num_eps = len(trace_data)
    masks_for_every_ep = []
    unfiltered_masks_for_every_ep = []
    for ep in range(num_eps):
        print("Episode {} in progress".format(ep))
        sub_dir = trace_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        masks_for_ep = []
        unfiltered_masks_for_ep = []
        for step in range(ep_len):
            image_loc = sub_dir + "trace_imgs/" + img_base_filename + str(step) + ".png"
            # image = load_img_and_convert_to_three_channels(image_loc)
            print("Starting to generate masks for step {} out of {}".format(step, ep_len - 1))
            image, unfiltered_masks, masks = run_segmentation_on_single_image(image_loc, sam_checkpoint, model_type)
            # masks = generate_and_filter_masks(image, sam_checkpoint, model_type)
            print("Finished generating masks for step {} out of {}".format(step, ep_len - 1))
            masks_for_ep.append(masks)
            unfiltered_masks_for_ep.append(unfiltered_masks)
        masks_for_every_ep.append(masks_for_ep)
        unfiltered_masks_for_every_ep.append(unfiltered_masks_for_ep)
        print("All masks made for this trace. Dumping it via a pickle now.")
        with open(sub_dir + model_type + "_results/filtered_masks.pkl", "wb") as f:
            pickle.dump(masks_for_ep, f)
        with open(sub_dir + model_type + "_results/unfiltered_masks.pkl", "wb") as g:
            pickle.dump(unfiltered_masks_for_ep, g)
    return masks_for_every_ep, unfiltered_masks_for_every_ep

def save_images_with_masks_and_events(masks_pkl_loc, trace_imgs_dir, trace_img_base_filename, events, dir_to_save_img):
    with open(masks_pkl_loc, "rb") as f:
        masks_for_ep = pickle.load(f)
    for i in range(len(masks_for_ep)):
        masks = masks_for_ep[i]
        event = events[i]
        image_loc = trace_imgs_dir + trace_img_base_filename + str(i) + ".png"
        image = load_img_and_convert_to_three_channels(image_loc)
        mask_img_filename = "masked_step_" + str(i) + ".png"
        path = dir_to_save_img + mask_img_filename
        save_image_with_masks(masks, image, path, event)
        print("Image saved for step {} out of {}".format(i, len(masks_for_ep) - 1))

def generate_event_labels_from_masks(trace_data, trace_dir, model_type, masks_for_ep_filename, img_base_filename, masks_for_every_ep=None):
    num_eps = len(trace_data)
    events_for_every_ep = []
    for ep in range(num_eps):
        print("Episode {} in progress".format(ep + 1))
        # Assuming that no event is observed in the initial state. This assumption should be dropped
        events_for_ep = [set()]
        sub_dir = trace_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        results_dir = sub_dir + model_type + "_results/" 
        masks_for_ep = None
        if masks_for_every_ep is None:
            masks_pkl_loc = results_dir + "filtered_masks.pkl"
            with open(masks_pkl_loc, "rb") as f:
                masks_for_ep = pickle.load(f)
        else:
            masks_for_ep = masks_for_every_ep[ep]
        # Still need to original image here to get the original vocab, can't use the masks alone. This worries me
        first_image_loc = sub_dir + "trace_imgs/" + img_base_filename + str(0) + ".png"
        first_image = load_img_and_convert_to_three_channels(first_image_loc)
        print("Step 0 snapshot loaded.")
        event_vocab = create_event_vocab(masks_for_ep[0], first_image)
        print("Event vocab created.")
        for step in range(1, ep_len):
            image_loc = sub_dir + "trace_imgs/" + img_base_filename + str(step) + ".png"
            image = load_img_and_convert_to_three_channels(image_loc)
            events = get_events_from_masks_in_state(event_vocab, masks_for_ep[step], image)
            print("Events gathered for step {}".format(step))
            events_for_ep.append(events)
        event_pkl_loc = results_dir + "events_updated.pkl"  
        with open(event_pkl_loc, "wb") as f:
            pickle.dump(events_for_ep, f)
        events_for_every_ep.append(events_for_ep)
    return events_for_every_ep
    
def inspect_events(events_pkl_loc, model_type):
    print("Events observed for the results from " + model_type)
    with open(events_pkl_loc, "rb") as f:
        events = pickle.load(f)
    print(events)

if __name__ == "__main__":
    events_observed = []
    traces_dir = "ww_trace/"
    trace_data_filename = "traces_data.pkl"
    img_base_filename = "env_step"
    sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    masks_for_ep_filename = "masks_" + model_type + ".pkl"
    filtered_masks_for_ep_fname = "filtered_masks.pkl"
    unfiltered_masks_for_ep_fname = "unfiltered_masks.pkl"

    with open(traces_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)
    num_eps = len(trace_data)

    # masks_for_every_ep = None
    # masks_for_every_ep, _ = generate_and_save_masks_for_eps(trace_data, traces_dir, sam_checkpoint, model_type, img_base_filename)

    # events_for_every_ep = generate_event_labels_from_masks(trace_data, traces_dir, model_type, masks_for_ep_filename, img_base_filename)

    # num_eps = len(trace_data)
    # for i in range(num_eps):
    #     trace_sub_dir = traces_dir + "trace_" + str(i) + "/"
    #     results_dir = trace_sub_dir + model_type + "_results/"
    #     masks_pkl_filename = "masks.pkl"
    #     trace_img_dir = trace_sub_dir + "trace_imgs/"
    #     masks_imgs_dir = results_dir + "masks_updated_imgs/" 
    #     events_ep_i = events_for_every_ep[i]
    #     save_images_with_masks_and_events(results_dir + "filtered_masks.pkl", trace_img_dir, img_base_filename, events_ep_i, masks_imgs_dir)

    for i in range(num_eps):
        trace_sub_dir = traces_dir + "trace_" + str(i) + "/"
        trace_img_dir = trace_sub_dir + "trace_imgs/"
        masks_pkl_loc = trace_sub_dir + model_type + "_results/" + unfiltered_masks_for_ep_fname
        # events_pkl_loc = trace_sub_dir + model_type + "_results/events.pkl"
        # inspect_events(events_pkl_loc, model_type)
        with open(masks_pkl_loc, "rb") as f:
            masks = pickle.load(f)
        for j in range(len(masks)):
            print("STEP {}".format(j))
            image_loc = trace_img_dir + img_base_filename + str(j) + ".png"
            image = load_img_and_convert_to_three_channels(image_loc)
            masks_step_j = masks[j]
            masks_step_j = add_masks_colours(masks_step_j, image)
            print("No of masks: {}".format(len(masks_step_j)))
            mask_areas = list(map(lambda mask: mask['area'], masks_step_j))
            print("Mask areas: " + str(mask_areas))
            colours = list(map(lambda mask: mask['colour'], masks_step_j))
            print("Mask colours: " + str(colours))