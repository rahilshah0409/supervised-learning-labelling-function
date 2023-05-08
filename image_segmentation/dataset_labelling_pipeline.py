import pickle
import sys
sys.path.insert(1, "segment_anything/")
from segment_with_sa import find_mask_colours, generate_and_filter_masks, get_event_occured, save_image_with_masks
from img_utils import load_img_and_convert_to_three_channels

def generate_and_save_masks_for_eps(trace_data, trace_dir, sam_checkpoint, model_type, img_base_filename):
    num_eps = len(trace_data)
    for ep in range(num_eps):
        print("Episode {} in progress".format(ep))
        sub_dir = trace_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        masks_for_ep = []
        for step in range(ep_len):
            image_loc = sub_dir + img_base_filename + str(step) + ".png"
            image = load_img_and_convert_to_three_channels(image_loc)
            print("Starting to generate masks for step {} out of {}".format(step, ep_len - 1))
            masks = generate_and_filter_masks(image, sam_checkpoint, model_type)
            print("Finished generating masks for step {} out of {}".format(step, ep_len - 1))
            masks_for_ep.append(masks)
        
        print("All masks made for this trace. Dumping it via a pickle now.")
        with open(sub_dir + masks_for_ep_filename, "wb") as f:
            pickle.dump(masks_for_ep, f)

def save_images_with_masks(masks_pkl_loc, trace_imgs_dir, trace_img_base_filename, events, dir_to_save_img):
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

def generate_event_labels_from_masks(trace_data, trace_dir, model_type, masks_for_ep_filename, img_base_filename):
    num_eps = len(trace_data)
    events_for_every_ep = []
    for ep in range(num_eps):
        print("Episode {} in progress".format(ep + 1))
        # Assuming that no event is observed in the initial state. This assumption should be dropped
        events_for_ep = [set()]
        sub_dir = trace_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        results_dir = sub_dir + model_type + "_results/" 
        masks_pkl_loc = results_dir + masks_for_ep_filename
        with open(masks_pkl_loc, "rb") as f:
            masks_for_ep = pickle.load(f)
        # Still need to original image here to get the original vocab, can't use the masks alone. This worries me
        first_image_loc = sub_dir + "trace_imgs/" + img_base_filename + str(0) + ".png"
        first_image = load_img_and_convert_to_three_channels(first_image_loc)
        print("Step 0 snapshot loaded.")
        _, _, event_vocab = find_mask_colours(masks_for_ep[0], first_image)
        print("Event vocab created.")
        for step in range(1, ep_len):
            image_loc = sub_dir + "trace_imgs/" + img_base_filename + str(step) + ".png"
            image = load_img_and_convert_to_three_channels(image_loc)
            events = get_event_occured(event_vocab, masks_for_ep[step], image)
            print("Events gathered for step {}".format(step))
            events_for_ep.append(events)
        # event_pkl_loc = results_dir + "events.pkl"  
        # with open(event_pkl_loc, "wb") as f:
        #     pickle.dump(events_for_ep, f)
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
    masks_for_ep_filename = "masks_vit_b_small_filter.pkl"
    # sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_b"

    with open(traces_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)

    # generate_and_save_masks_for_eps(trace_data, trace_dir, sam_checkpoint, model_type, img_base_filename)

    events_for_every_ep = generate_event_labels_from_masks(trace_data, traces_dir, model_type, masks_for_ep_filename, img_base_filename)

    num_eps = len(trace_data)
    for i in range(num_eps):
        trace_sub_dir = traces_dir + "trace_" + str(i) + "/"
        results_dir = trace_sub_dir + model_type + "_results/"
        masks_pkl_filename = "masks_" + model_type + "_small_filter.pkl"
        trace_img_dir = trace_sub_dir + "trace_imgs/"
        masks_imgs_dir = results_dir + "masks_imgs/" 
        events_ep_i = events_for_every_ep[i]
        save_images_with_masks(results_dir + masks_pkl_filename, trace_img_dir, img_base_filename, events_ep_i, masks_imgs_dir)

    # num_eps = len(trace_data)
    # for i in range(num_eps):
    #     trace_sub_dir = traces_dir + "trace_" + str(i) + "/"
    #     events_pkl_loc = trace_sub_dir + model_type + "_results/events.pkl"
    #     inspect_events(events_pkl_loc, model_type)