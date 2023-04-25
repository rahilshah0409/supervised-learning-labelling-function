import sys
sys.path.insert(1, "..")
from img_utils import load_img_and_convert_to_three_channels
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color= []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))

def show_image(img, masks=None):
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    if masks is not None:
        show_anns(masks)
    plt.axis('off')
    plt.show()

def generate_masks(image, sam_checkpoint, model_type):
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks

def inspect_masks(masks):
    print("The number of masks extracted is {}".format(len(masks)))
    first_mask = masks[0]
    print(first_mask['segmentation'])
    print(first_mask['bbox'])
    print(first_mask['area'])
    print(first_mask['predicted_iou'])
    print(first_mask['crop_box'])
    print(first_mask['point_coords'])

if __name__ == "__main__":
    eg_img_path = "../waterworld_imgs/example.png"
    image = load_img_and_convert_to_three_channels(eg_img_path)
    # print(image)
    sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type="vit_h"
    masks = generate_masks(image, sam_checkpoint, model_type)
    show_image(image, masks)
    inspect_masks(masks)