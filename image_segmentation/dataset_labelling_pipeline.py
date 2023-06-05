import pickle
import sys
sys.path.insert(1, "../")
from image_segmentation.segment_with_sa import create_event_vocab, get_events_from_masks_in_state, save_image_with_masks, run_segmentation_on_single_image, add_masks_colours
from image_segmentation.img_utils import load_img_and_convert_to_three_channels

def generate_and_save_masks_for_eps(trace_data, trace_dir, sam_checkpoint, model_type, img_base_filename, filtered_masks_fname_base):
    num_eps = len(trace_data)
    masks_for_every_ep = []
    # unfiltered_masks_for_every_ep = []
    for ep in range(num_eps):
        print("EPISODE {}: STARTING MASK GENERATION".format(ep))
        sub_dir = trace_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        masks_for_ep = []
        # unfiltered_masks_for_ep = []
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
            # unfiltered_masks_for_ep.append(unfiltered_masks)
        masks_for_every_ep.append(masks_for_ep)
        # unfiltered_masks_for_every_ep.append(unfiltered_masks_for_ep)
        print("EPISODE {}: FINISHED MASK GENERATION.".format(ep))
        # results_dir = sub_dir
        # filtered_masks_path = results_dir + filtered_masks_fname_base + str(step) + ".pkl"
        # with open(filtered_masks_path, "wb") as f:
        #     pickle.dump(masks_for_ep, f)
        # unfiltered_masks_path = results_dir + unfiltered_masks_fname
        # with open(unfiltered_masks_path, "wb") as g:
        #     pickle.dump(unfiltered_masks_for_ep, g)
    return masks_for_every_ep

def save_images_with_masks_and_events(ep_len, masks_pkl_dir, masks_base_fname, trace_imgs_dir, trace_img_base_filename, events, dir_to_save_img):
    # Get the masks you want to illustrate
    with open(masks_pkl_dir + "filtered_masks_2.pkl", "rb") as f:
        masks_for_ep = pickle.load(f)
    for step in range(ep_len):
        masks = masks_for_ep[step]
        event = events[step]
        image_loc = trace_imgs_dir + trace_img_base_filename + str(step) + ".png"
        image = load_img_and_convert_to_three_channels(image_loc)
        mask_img_filename = masks_base_fname + str(step) + ".png"
        path = dir_to_save_img + mask_img_filename
        save_image_with_masks(masks, image, path, event)
        # print("Image saved for step {} out of {}".format(i, len(masks_for_ep) - 1))

def generate_event_labels_from_masks(trace_data, trace_dir, model_type, masks_fname_base, img_base_fname, events_fname, masks_for_every_ep=None):
    num_eps = len(trace_data)
    events_for_every_ep = []
    # This variable keeps track of every event observed when generating this dataset, hoping that we get full coverage when we generate the dataset
    events_observed = set()
    for ep in range(num_eps):
        print("Episode {} in progress".format(ep + 1))
        # Assuming that no event is observed in the initial state. This assumption should be dropped
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

        # Still need to original image here to get the original vocab, can't use the masks alone. This worries me
        # Create the event vocab with the first frame
        first_image_loc = trace_img_dir + img_base_fname + str(0) + ".png"
        first_image = load_img_and_convert_to_three_channels(first_image_loc)
        print("Step 0 snapshot loaded.")
        event_vocab, past_observed_events, no_of_expected_masks = create_event_vocab(first_masks, first_image)
        print("Event vocab created.")
        for step in range(1, ep_len):
            image_loc = trace_img_dir + img_base_fname + str(step) + ".png"
            image = load_img_and_convert_to_three_channels(image_loc)
            # Get the events at every time step of the episode in question and record this
            masks_i = None
            if masks_for_ep is not None:
                masks_i = masks_for_ep[step]
            else:
                masks_i_loc = sub_dir + masks_fname_base + str(step) + ".pkl"
                with open(masks_i_loc, "rb") as f:
                    masks_i = pickle.load(f)
            events, past_observed_events = get_events_from_masks_in_state(event_vocab, masks_i, image, past_observed_events, no_of_expected_masks)
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
    traces_dir = "single_trace_experimentation/ww_trace_black/"
    trace_data_filename = "traces_data.pkl"
    img_base_fname = "env_step"
    sam_checkpoint = "/vol/bitbucket/ras19/fyp/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    masks_for_ep_filename = "masks.pkl"
    filtered_masks_for_ep_fname = "filtered_masks.pkl"
    unfiltered_masks_for_ep_fname = "unfiltered_masks.pkl"
    events_fname = "new_events.pkl"
    masks_base_fname = "masked_step"

    with open(traces_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)
    num_eps = len(trace_data)

    masks_for_every_ep = None
    # masks_for_every_ep, _ = generate_and_save_masks_for_eps(trace_data, traces_dir, sam_checkpoint, model_type, img_base_fname, masks_base_fname)

    events_for_every_ep, events_observed = generate_event_labels_from_masks(trace_data, traces_dir, model_type, masks_base_fname, img_base_fname, events_fname, masks_for_every_ep)

    for i in range(num_eps):
        ep_len = trace_data[i]["length"]
        trace_sub_dir = traces_dir + "trace_" + str(i) + "/"
        results_dir = trace_sub_dir + model_type + "_results/"
        masks_pkl_filename = "filtered_masks_2.pkl"
        trace_img_dir = trace_sub_dir + "trace_imgs/"
        masks_imgs_dir = results_dir + "new_masks_imgs/" 
        # event_pkl_loc = results_dir + events_fname
        # with open(event_pkl_loc, "rb") as f:
        #     events_ep_i= pickle.load(f)
        events_ep_i = events_for_every_ep[i]
        save_images_with_masks_and_events(ep_len, trace_sub_dir, masks_base_fname, trace_img_dir, img_base_fname, events_ep_i, masks_imgs_dir)

    # for i in range(num_eps):
    #     trace_sub_dir = traces_dir + "trace_" + str(i) + "/"
    #     trace_img_dir = trace_sub_dir + "trace_imgs/"
    #     masks_pkl_loc = trace_sub_dir + model_type + "_results/" + unfiltered_masks_for_ep_fname
    #     # events_pkl_loc = trace_sub_dir + model_type + "_results/events.pkl"
    #     # inspect_events(events_pkl_loc, model_type)
    #     with open(masks_pkl_loc, "rb") as f:
    #         masks = pickle.load(f)
    #     for j in range(len(masks)):
    #         print("STEP {}".format(j))
    #         image_loc = trace_img_dir + img_base_filename + str(j) + ".png"
    #         image = load_img_and_convert_to_three_channels(image_loc)
    #         masks_step_j = masks[j]
    #         masks_step_j = add_masks_colours(masks_step_j, image)
    #         print("No of masks: {}".format(len(masks_step_j)))
    #         mask_areas = list(map(lambda mask: mask['area'], masks_step_j))
    #         print("Mask areas: " + str(mask_areas))
    #         colours = list(map(lambda mask: mask['colour'], masks_step_j))
    #         print("Mask colours: " + str(colours))