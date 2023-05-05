import pickle
import sys
sys.path.insert(1, "segment_anything/")
from segment_with_sa import find_mask_colours, generate_and_filter_masks, get_event_occured
from img_utils import load_img_and_convert_to_three_channels

def generate_and_save_masks_for_eps(trace_data, trace_dir, sam_checkpoint, model_type, img_base_filename):
    num_eps = len(trace_data)
    for ep in num_eps:
        print("Episode {} in progress".format(ep))
        sub_dir = trace_dir + "trace_" + ep + "/"
        ep_len = trace_data[ep]["length"]
        masks_for_ep = []
        for step in range(ep_len):
            image = load_img_and_convert_to_three_channels(sub_dir + img_base_filename + str(step))
            print("Starting to generate masks for step {} out of {}".format(step, ep_len - 1))
            masks = generate_and_filter_masks(image, sam_checkpoint, model_type)
            print("Finished generating masks for step {} out of {}".format(step, ep_len - 1))
            masks_for_ep.append(masks)
        
        print("All masks made for this trace. Dumping it via a pickle now.")
        with open(sub_dir + masks_for_ep_filename, "wb") as f:
            pickle.dump(masks_for_ep, f)

def generate_event_labels_from_masks(trace_data, trace_dir, masks_for_ep_filename, img_base_filename):
    num_eps = len(trace_data)
    for ep in num_eps:
        print("Episode {} in progress".format(ep + 1))
        events_for_ep = []
        sub_dir = trace_dir + "trace_" + ep + "/"
        ep_len = trace_data[ep]["length"]
        with open(sub_dir + masks_for_ep_filename, "rb") as f:
            masks_for_ep = pickle.load(f)
        # Still need to original image here to get the original vocab, can't use the masks alone. This worries me
        image = load_img_and_convert_to_three_channels(sub_dir + img_base_filename + str(0))
        print("Step snapshot loaded.")
        event_vocab = find_mask_colours(masks_for_ep[0], image)
        print("Event vocab created.")
        for step in range(1, ep_len):
            image = load_img_and_convert_to_three_channels(sub_dir + img_base_filename + str(step))
            events = get_event_occured(event_vocab, masks_for_ep[step], image)
            print("Events gathered for step {}".format(step))
            events_for_ep.append(events)

if __name__ == "__main__":
    events_observed = []
    trace_dir = "ww_trace/"
    trace_data_filename = "traces_data.pkl"
    img_base_filename = "env_step"
    masks_for_ep_filename = "masks_for_ep.pkl"
    sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    with open(trace_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)

    generate_and_save_masks_for_eps(trace_data, trace_dir, sam_checkpoint, model_type, img_base_filename)

    # generate_event_labels_from_masks(trace_data, trace_dir, masks_for_ep_filename, img_base_filename)