from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
# The two paths below are hardcoded from my local machine and hence will be useless when working on lab machines
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/image-segmentation-experimentation/image_segmentation")
sys.path.insert(1, "/home/rahilshah/Documents/Year4/FYP/image-segmentation-experimentation/image_generation")
sys.path.insert(1, "..")
from img_utils import load_img_and_convert_to_three_channels
# from generate_imgs import ball_area

ball_rad = 15 # default radius, be wary if this changes
ball_area = math.pi * (ball_rad ** 2)

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
            img[:, :, i] = color_mask[i]
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
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        mask_on_image = np.dstack((img, m * 0.35))
        ax.imshow(mask_on_image)


def show_image(img, masks=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    if masks is not None:
        show_anns(masks)
    plt.axis('off')
    plt.show()


def save_image_with_axis(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('on')
    plt.savefig("../eg_ww_img/example_with_axis.png")


def generate_and_save_masks(image, sam_checkpoint, model_type):
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masks = filter_small_masks(masks, 100)
    masks, centres = find_mask_colour(masks, image)
    with open("../eg_ww_img/masks.pkl", "wb") as f:
        pickle.dump(masks, f)
    return masks, centres


def inspect_masks(masks):
    print("The number of masks extracted is {}".format(len(masks)))
    first_mask = masks[0]
    print(first_mask['segmentation'])
    print(first_mask['bbox'])
    print(first_mask['area'])
    print(first_mask['predicted_iou'])
    print(first_mask['crop_box'])
    print(first_mask['point_coords'])


def filter_small_masks(masks, uncertainty):
    filtered_masks = list(filter(lambda m: m['area'] >= ball_area - uncertainty and m['area'] <= ball_area + uncertainty, masks))
    return filtered_masks


# Either can look at the individual pixels or look at the centre of the box that surrounds the box. Prefer the latter. Assuming x increases as you go to the right and y increases as you go down. Pretty sure the flooring doesn't matter in this function, it is just so I can get whole numbers that I can then use as co-ords
def find_mask_colour(masks, image):
    plt.figure(figsize=(20, 20))
    centres = []
    for mask in masks:
        [x, y, width, height] = mask['bbox']
        x_centre = x + math.floor(width / 2)
        y_centre = y + math.floor(height / 2)
        centres.append((x_centre, y_centre))
        bgr_colour = image[y_centre, x_centre, :]
        mask['colour'] = bgr_colour
    return masks, centres


if __name__ == "__main__":
    eg_img_path = "../eg_ww_img/example.png"
    image = load_img_and_convert_to_three_channels(eg_img_path)
    print("Loaded and converted image to 3 channels")
    sam_checkpoint = "/vol/bitbucket/ras19/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    # print("Generating masks using Segment Anything and saving them for later use")
    # masks, centres = generate_and_save_masks(image, sam_checkpoint, model_type)
    # print("Completed mask generation and masks are saved")
    with open("../eg_ww_img/masks.pkl", "rb") as f:
        masks = pickle.load(f)
    # show_image(image, masks)
    # save_image_with_masks(image, masks)
    # print("Inspecting masks produced")
    # inspect_masks(masks)
