import easygui
from image import normalize, show_img
import unicodedata
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename
from image import show_img, show_no_wait_img
import cv2

MINIMUM_AREA = 500


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


def calculate_area_thresh(contours, minimum_area=0):
    contours_new = []
    for contour in contours:
        tmp_area = cv2.contourArea(contour)
        if tmp_area>minimum_area:
            contours_new += [contour,]
    return contours_new


def contour_of_not_borders(contours, maxH, maxW):
    maxH -= 1
    maxW -= 1
    contours_new = []
    for contour in contours:
        flag = True
        for points in contour:
            point = points[0]
            if point[0] == 0 or point[1] == 0 or point[0] == maxH or point[1] == maxW:
                flag = False
                break
        if flag:
            contours_new += [contour, ]
    return contours_new


def contour_bounding_box(contours, maxH, maxW):
    contours_new = []
    for contour in contours:
        x,y,width, height = cv2.boundingRect(contour)
        if not(x == 0 or y == 0 or x+width == maxW or y+height == maxH):
            contours_new += [contour, ]
    return contours_new


def ratio_borders(contours, thresh):
    contours_new = []
    for contour in contours:
        center,(width,height),angle = cv2.minAreaRect(contour)
        if not(width/height>thresh or height/width>thresh):
            contours_new += [contour, ]
    return contours_new

def main():
    img_path = "/Users/idamaruotto/Downloads/Mammotest/acanthocyte1.jpg" #img_path_read()
    image = cv2.imread(img_path)

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #objects must be white over a black bg
    show_no_wait_img(img, 'Original Image')

    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    maxW = image.shape[0]
    maxH = image.shape[1]

    #selecting internal contours
    internal_contours = contour_bounding_box(contours, maxW, maxH)
    #removing small elements
    internal_contours = calculate_area_thresh(internal_contours, MINIMUM_AREA)
    #separate classes
    ratio_contours = ratio_borders(internal_contours, 1.6)

    #show_no_wait_img(cv2.drawContours(image.copy(), contours, contourIdx=-1, color=(0, 0, 0), thickness=2), 'All')
    img1 = image.copy()
    img1 = cv2.drawContours(img1, internal_contours, contourIdx=-1, color=(0, 0, 0), thickness=8)
    img1 = cv2.drawContours(img1, ratio_contours, contourIdx=-1, color=(255, 0, 0), thickness=2)
    show_img(img1, 'Internal')
    print('internal: ', len(internal_contours), '\t separating or double: ', len(ratio_contours))


if __name__ == '__main__':
    main()
