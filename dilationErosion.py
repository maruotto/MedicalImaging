import easygui
from matplotlib import pyplot as plt

from image import normalize, show_img
import unicodedata
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename
from image import show_img, show_no_wait_img
import cv2
import math
import numpy as np
from binary_img_lib import contour_bounding_box, calculate_area_thresh
from sklearn.cluster import AgglomerativeClustering

from image import histogram

d_slider_max = 40

def on_trackbar(val):
    global image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (val, val))
    # Using cv2.erode() method
    eroded = cv2.erode(image, kernel)

    # Displaying the image
    show_no_wait_img(two_images(image, eroded), 'Images')
    cv2.waitKey()


def two_images(image1, image2, horizontal = True):
    """
    the two images must have the same shape
    returns an image that can be shown with cv2.imshow()
    """
    if horizontal:
        stack = np.hstack((image1, image2)) #np.concatenate((image1, image2), axis=1)
    else: np.vstack((image1, image2)) #np.concatenate((image1, image2), axis=0)

    return stack

def img_path_read():
    tk().withdraw()
    img_path = askopenfilename(message="Choose a file to open", initialdir=r"/Users/idamaruotto/Downloads/Mammotest")
    print(img_path)
    #uni_code = easygui.fileopenbox(msg="Choose a file to open",
    #                               default=r"/Users/idamaruotto/Downloads/Mammotest")
    img_path = unicodedata.normalize('NFKD', img_path).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')
    #img_path = "/Users/idamaruotto/Downloads/Segmentaziones/dataset/train/00001_p0_patch0.tif"
    return img_path


def main():
    global image
    img_path = img_path_read()
    image = cv2.imread(img_path)
    cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Circle of diameter', "Images", 1, d_slider_max, on_trackbar)

    # create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # Using cv2.erode() method
    eroded = cv2.erode(image, kernel)

    # Displaying the image
    show_no_wait_img(two_images(image, eroded), 'Images')


if __name__ == '__main__':
    main()
    cv2.waitKey()

