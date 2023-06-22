import sys
sys.path.insert(1, "..")
from img_utils import load_img_and_convert_to_three_channels
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    dir_path = "../eg_ww_img/"
    orig_img_name = "example.png"

    eg_img_path = dir_path + orig_img_name
    image = load_img_and_convert_to_three_channels(eg_img_path)
    print("Loaded and converted image to 3 channels")
  
    circles = cv.HoughCircles(image,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
      # draw the outer circle
      cv.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
      # draw the center of the circle
      cv.circle(image,(i[0],i[1]),2,(0,0,0),3)

    cv.imshow('detected circles',image)