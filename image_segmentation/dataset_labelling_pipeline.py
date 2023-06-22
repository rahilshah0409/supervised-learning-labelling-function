import pickle
import sys
sys.path.insert(1, "../")
from image_segmentation.segment_with_sa import create_event_vocab, get_events_from_masks_in_state, save_image_with_masks, run_segmentation_on_single_image, add_masks_colours
from image_segmentation.img_utils import load_img_and_convert_to_three_channels

# Use Segment Anything to identify objects in all state renderings in a dataset
def generate_and_save_masks_for_eps(trace_data, trace_dir, sam_checkpoint, model_type, img_base_filename, filtered_masks_fname_base):
    num_eps = len(trace_data)
    masks_for_every_ep = []
    for ep in range(num_eps):
        print("EPISODE {}: STARTING MASK GENERATION".format(ep))
        sub_dir = trace_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        masks_for_ep = []
        for step in range(ep_len):
            image_loc = sub_dir + img_base_filename + str(step) + ".png"
            print("Starting to generate masks for step {} out of {}".format(step, ep_len - 1))
            _, unfiltered_masks, masks = run_segmentation_on_single_image(image_loc, sam_checkpoint, model_type)
            print("Finished generating masks")
            print("Saving masks for this step")
            results_dir = sub_dir
            filtered_masks_path = results_dir + filtered_masks_fname_base + str(step) + ".pkl"
            with open(filtered_masks_path, "wb") as f:
                pickle.dump(masks, f)
            masks_for_ep.append(masks)
        masks_for_every_ep.append(masks_for_ep)
        print("EPISODE {}: FINISHED MASK GENERATION.".format(ep))
    return masks_for_every_ep

# Save renderings of the environment, together with the objects that were detected in that rendering and the events that were labelled.
def save_images_with_masks_and_events(ep_len, masks_pkl_dir, masks_base_fname, trace_imgs_dir, trace_img_base_filename, events, dir_to_save_img):
    # Get the masks you want to illustrate
    for step in range(ep_len):
        with open(masks_pkl_dir + "masks" + str(step) + ".pkl", "rb") as f:
            masks = pickle.load(f)
        # masks = masks_for_ep[step]
        event = events[step]
        image_loc = trace_imgs_dir + trace_img_base_filename + str(step) + ".png"
        image = load_img_and_convert_to_three_channels(image_loc)
        mask_img_filename = masks_base_fname + str(step) + ".png"
        path = dir_to_save_img + mask_img_filename
        save_image_with_masks(masks, image, path, event)
        # print("Image saved for step {} out of {}".format(i, len(masks_for_ep) - 1))

# Creates the event labels for the dataset, given that the objects have been identified with Segment Anything
def generate_event_labels_from_masks(trace_data, trace_dir, model_type, masks_fname_base, img_base_fname, events_fname, masks_for_every_ep=None):
    num_eps = len(trace_data)
    events_for_every_ep = []
    events_observed = set()
    for ep in range(num_eps):
        print("Episode {} in progress".format(ep + 1))
        events_for_ep = [set()]
        ep_len = trace_data[ep]["length"]
        sub_dir = trace_dir + "trace_" + str(ep) + "/"
        trace_img_dir = sub_dir
        results_dir = sub_dir
        # Get the masks (either masks for all episodes are given in the arguments or they need to be loaded from the pickle object)
        masks_for_ep = None
        if masks_for_every_ep is not None:
            masks_for_ep = masks_for_every_ep[ep]
        
        first_masks = None
        if masks_for_ep is not None:
            first_masks = masks_for_ep[0]
        else:
            first_masks_loc = sub_dir + masks_fname_base + str(0) + ".pkl"
            with open(first_masks_loc, "rb") as f:
                first_masks = pickle.load(f)

        # Create the event configuration with the first frame
        first_image_loc = trace_img_dir + img_base_fname + str(0) + ".png"
        first_image = load_img_and_convert_to_three_channels(first_image_loc)
        print("Step 0 snapshot loaded.")
        event_vocab, freq_of_colours, ball_centres, common_obj_size, expected_radius = create_event_vocab(first_masks, first_image)
        no_of_expected_objs = sum(list(freq_of_colours.values()))
        print("Event vocab created.")
        for step in range(1, ep_len):
            image_loc = trace_img_dir + img_base_fname + str(step) + ".png"
            image = load_img_and_convert_to_three_channels(image_loc)
            masks_i = None
            if masks_for_ep is not None:
                masks_i = masks_for_ep[step]
            else:
                masks_i_loc = sub_dir + masks_fname_base + str(step) + ".pkl"
                with open(masks_i_loc, "rb") as f:
                    masks_i = pickle.load(f)
            events = get_events_from_masks_in_state(event_vocab, masks_i, image, freq_of_colours, no_of_expected_objs, common_obj_size, ball_centres, expected_radius)
            print("Events gathered for step {}".format(step))
            events_observed.update(events)
            events_for_ep.append(events)
        # Save all of the events discovered for this episode
        event_pkl_loc = results_dir + events_fname  
        with open(event_pkl_loc, "wb") as f:
            pickle.dump(events_for_ep, f)
        events_for_every_ep.append(events_for_ep)
    return events_for_every_ep, events_observed
    
def inspect_events(events_pkl_loc, model_type):
    print("Events observed for the results from " + model_type)
    with open(events_pkl_loc, "rb") as f:
        events = pickle.load(f)
    print(events)

if __name__ == "__main__":
    events_observed = []
    traces_dir = "single_trace_experimentation/trace_from_train/"
    trace_data_filename = "traces_data.pkl"
    img_base_fname = "step"
    sam_checkpoint = "/vol/bitbucket/ras19/fyp/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    masks_for_ep_filename = "masks.pkl"
    filtered_masks_for_ep_fname = "filtered_masks.pkl"
    unfiltered_masks_for_ep_fname = "unfiltered_masks.pkl"
    events_fname = "final_final_events.pkl"
    masks_img_fname_base = "masked_step"
    masks_pkl_fname_base = "masks"

    with open(traces_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)
    num_eps = len(trace_data)

    masks_for_every_ep = None
    masks_for_every_ep, _ = generate_and_save_masks_for_eps(trace_data, traces_dir, sam_checkpoint, model_type, img_base_fname, masks_pkl_fname_base)

    events_for_every_ep, events_observed = generate_event_labels_from_masks(trace_data, traces_dir, model_type, masks_pkl_fname_base, img_base_fname, events_fname, masks_for_every_ep)

    for i in range(1):
        ep_len = trace_data[i]["length"]
        trace_sub_dir = traces_dir + "trace_" + str(i) + "/"
        results_dir = trace_sub_dir
        trace_img_dir = trace_sub_dir
        masks_imgs_dir = results_dir + "final_final_masks_imgs/" 
        event_pkl_loc = results_dir + events_fname
        with open(event_pkl_loc, "rb") as f:
            events_ep_i= pickle.load(f)
        save_images_with_masks_and_events(ep_len, trace_sub_dir, masks_img_fname_base, trace_img_dir, img_base_fname, events_ep_i, masks_imgs_dir)