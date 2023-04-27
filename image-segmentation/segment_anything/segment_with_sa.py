import sys
sys.path.insert(1, "..")
from img_utils import load_img_and_convert_to_three_channels
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def save_image_with_masks(img_original, masks):
    print("Dimensions of original image")
    print(img_original.shape[0])
    print(img_original.shape[1])
    print(img_original.shape[2])
    original_image = Image.fromarray(img_original)
    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        img_with_one_mask = np.dstack((img, m * 0.35))
        print("Dimensions of image overlayed with a mask")
        print(img_with_one_mask.shape[0])
        print(img_with_one_mask.shape[1])
        print(img_with_one_mask.shape[2])
        # original_image.paste(img_with_one_mask)
    # original_image.save("../saved_images/segment_anything.jpg")

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
        mask_on_image = np.dstack((img, m * 0.35))   
        ax.imshow(mask_on_image)

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
    print("Loaded and converted image to 3 channel")
    sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type="vit_h"
    print("Generating masks using Segment Anything")
    masks = generate_masks(image, sam_checkpoint, model_type)
    print("Completed mask generation")
    # show_image(image, masks)
    save_image_with_masks(image, masks)
    print("Inspecting masks produced")
    # inspect_masks(masks)