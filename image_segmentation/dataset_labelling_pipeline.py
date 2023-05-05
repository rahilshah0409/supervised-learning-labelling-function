import pickle
import sys
sys.path.insert(1, "segment_anything/")
from image_segmentation.segment_anything.segment_with_sa import find_mask_colours, generate_and_filter_masks, get_event_occured
from img_utils import load_img_and_convert_to_three_channels

if __name__ == "__main__":
    events_observed = []
    trace_dir = "ww_trace/"
    trace_data_filename = "traces_data.pkl"
    img_base_filename = "env_step"
    masks_for_ep_filename = "masks_for_ep.pkl"

    with open(trace_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)

    num_eps = len(trace_data)
    for ep in num_eps:
        sub_dir = trace_dir + "trace_" + ep + "/"
        ep_len = trace_data[ep]["length"]
        sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        masks_for_ep = []
        for step in range(ep_len):
            image = load_img_and_convert_to_three_channels(sub_dir + img_base_filename + str(step))
            masks = generate_and_filter_masks(image, sam_checkpoint, model_type)
            masks_for_ep.append(masks)
        
        with open(sub_dir + masks_for_ep_filename, "wb") as f:
            pickle.dump(masks_for_ep, f)

    for ep in num_eps:
        events_for_ep = []
        sub_dir = trace_dir + "trace_" + ep + "/"
        ep_len = trace_data[ep]["length"]
        with open(sub_dir + masks_for_ep_filename, "rb") as f:
            masks_for_ep = pickle.load(f)
        # Still need to original image here to get the original vocab, can't use the masks alone. This worries me
        image = load_img_and_convert_to_three_channels(sub_dir + img_base_filename + str(0))
        event_vocab = find_mask_colours(masks_for_ep[0], image)
        for step in range(1, ep_len):
            image = load_img_and_convert_to_three_channels(sub_dir + img_base_filename + str(step))
            events = get_event_occured(event_vocab, masks_for_ep[step], image)
            events_for_ep.append(events)