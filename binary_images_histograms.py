import easygui
from image import normalize, show_img
import unicodedata
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename
from image import show_img, show_no_wait_img
import cv2
import math
import numpy as np
from binary_img_lib import contour_bounding_box, calculate_area_thresh
from image import histogram

MINIMUM_AREA = 5


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


def evaluate_areas(contours):
    areas = []
    for contour in contours:
        tmp_area = cv2.contourArea(contour)
        areas += [tmp_area,]
    return areas

def evaluate_perimeters(contours):
    perimeters = []
    for contour in contours:
        tmp_perimeters = cv2.arcLength(contour,True)
        perimeters += [tmp_perimeters,]
    return perimeters

def evaluate_C(contours):
    perimeters = evaluate_perimeters(contours)
    areas = evaluate_areas(contours)
    C=[]
    for i in range(len(contours)):
        if perimeters[i] == 0 and areas[i] == 0:
            return None
        c = 4*math.pi*areas[i]
        c = c/math.pow(perimeters[i],2)
        C += [c,]
    return C


def evaluate_C_ratio(areas, perimeters):
    C = []
    for i in range(len(areas)):
        if perimeters[i] == 0 and areas[i] == 0:
            return None
        c = 4 * math.pi * areas[i]
        c = c / math.pow(perimeters[i], 2)
        C += [c, ]
    return C

def main():
    img_path = "/Users/idamaruotto/Downloads/Mammotest/acanthocyte1.jpg"
    image = cv2.imread(img_path)

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #objects must be white over a black bg
    show_no_wait_img(img, 'Original Image')

    contours, _ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    maxW = image.shape[0]
    maxH = image.shape[1]
    # selecting internal contours
    internal_contours = contour_bounding_box(contours, maxW, maxH)
    # removing small elements
    internal_contours = calculate_area_thresh(internal_contours, MINIMUM_AREA)

    C = evaluate_C(internal_contours)
    hist = np.histogram(np.array(C))
    histogram(np.array(C))


if __name__ == '__main__':
    main()
