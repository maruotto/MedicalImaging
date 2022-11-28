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


def hierarchical_clustering_algorithm(data):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    data = data.reshape(-1, 1)
    labels = hierarchical_cluster.fit_predict(data)
    return labels


def evaluate_ratio(contours):
    """For each contour calculates the ratio of the bounding box to separate elongated or double
        cells from others
        :returns the list of contours which the corresponding bounding """
    ratio = []
    for contour in contours:
        center,(width,height),angle = cv2.minAreaRect(contour)
        if (height>width):
            ratio += [width/height, ]
        else: ratio += [height/width, ]
    return ratio


def main():
    img_path = "/Users/idamaruotto/Downloads/Mammotest/acanthocyte1.jpg"
    image = cv2.imread(img_path)

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #objects must be white over a black bg
    #show_no_wait_img(img, 'Original Image')

    contours, _ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    maxW = image.shape[0]
    maxH = image.shape[1]
    # selecting internal contours
    internal_contours = contour_bounding_box(contours, maxW, maxH)
    # removing small elements
    internal_contours = calculate_area_thresh(internal_contours, MINIMUM_AREA)

    C = np.array(evaluate_C(internal_contours)) # they're at most 1 and at least 0 -> percentage of circularity
    #hist = np.histogram(np.array(C), bins=1, normed=True)
    print(C)

    plt.figure(1)
    plt.hist(C, bins=100)
    #plt hist uses these two
    counts, bins = np.histogram(C, bins=100)
    plt.stairs(counts, bins)

    plt.suptitle('Compactness')

    plt.figure(2)
    ratio = np.array(evaluate_ratio(internal_contours))
    plt.suptitle('Ratio smaller/longest')
    plt.hist(ratio, bins=100)
    #plt.show()

    dst=cv2.Laplacian(ratio, ddepth=-1)
    print(dst)
    plt.figure(3)
    plt.suptitle('Ratio smaller/longest with laplacian')
    plt.hist(dst, bins=100)

    labels_laplacian = hierarchical_clustering_algorithm(dst)
    labels_normal = hierarchical_clustering_algorithm(ratio)

    img1 = image.copy()
    ratio_contours_laplacian = [internal_contours[i] for i in range(len(internal_contours)) if labels_laplacian[i] == 1]
    ratio_contours_normal = [internal_contours[i] for i in range(len(internal_contours)) if labels_normal[i] == 1]
    img1 = cv2.drawContours(img1, ratio_contours_laplacian, contourIdx=-1, color=(150, 0, 0), thickness=8)
    img1 = cv2.drawContours(img1, ratio_contours_normal, contourIdx=-1, color=(0, 150, 0), thickness=2)
    show_no_wait_img(img1, 'Bigger cells')

    dst_circularity = cv2.Laplacian(C, ddepth=-1)
    labels_laplacian_circularity = hierarchical_clustering_algorithm(dst_circularity)
    labels_normal_circularity= hierarchical_clustering_algorithm(C)

    img2 = image.copy()

    circularity_laplacian = [internal_contours[i] for i in range(len(internal_contours)) if labels_laplacian_circularity[i] == 1]
    circularity_normal = [internal_contours[i] for i in range(len(internal_contours)) if labels_normal_circularity[i] == 1]
    img2 = cv2.drawContours(img2, circularity_laplacian, contourIdx=-1, color=(150, 0, 0), thickness=8)
    img2 = cv2.drawContours(img2, circularity_normal, contourIdx=-1, color=(0, 150, 0), thickness=2)
    show_no_wait_img(img2, 'Compactness')


    ##Scatter plot


if __name__ == '__main__':
    main()
    cv2.waitKey()
    plt.show()

