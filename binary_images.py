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
    """For each contour calculates the area
    :returns the list of contours that enclose an area bigger than minimum_area"""

    contours_new = []
    for contour in contours:
        tmp_area = cv2.contourArea(contour)
        if tmp_area>minimum_area:
            contours_new += [contour,]
    return contours_new


def contour_of_not_borders(contours, maxH, maxW):
    """Check if the contours touch the borders
        :returns the list of contours that do not have any of the contour point on an edge"""

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
    """For each contour calculates the bounding box (parallel to axes)
        :returns the list of contours which the corresponding bounding
        box does not have any side overlapping to a side of the image"""

    contours_new = []
    for contour in contours:
        x,y,width, height = cv2.boundingRect(contour)
        if not(x == 0 or y == 0 or x+width == maxW or y+height == maxH):
            contours_new += [contour, ]
    return contours_new


def ratio_borders(contours, thresh):
    """For each contour calculates the ratio of the bounding box to separate elongated or double
        cells from others
        :returns the list of contours which the corresponding bounding """
    contours_new = []
    for contour in contours:
        center,(width,height),angle = cv2.minAreaRect(contour)
        if (height/width>thresh or width/height>thresh):
            contours_new += [contour, ]
    return contours_new

def main():
    img_path = "/Users/idamaruotto/Downloads/Mammotest/acanthocyte1.jpg" #img_path_read() #macos users - static path
    image = cv2.imread(img_path) #import image

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #transfrom to grey level - single channel

    # to make the find contours work
    # objects must be white over a black bg -> THRESH_BINARY_INV
    # using THRESH_OTSU algorithm
    _, img = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    show_no_wait_img(img, 'Original Image') #my function to show images - is just cv2.imshow()

    # find only external contours with no chain approximation
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)


    #selecting internal contours
    internal_contours = contour_bounding_box(contours, image.shape[0], image.shape[1])
    #removing small elements
    internal_contours = calculate_area_thresh(internal_contours, MINIMUM_AREA)
    #separate classes
    ratio_contours = ratio_borders(internal_contours, 1.6)

    # show_no_wait_img(cv2.drawContours(image.copy(), contours, contourIdx=-1, color=(0, 0, 0), thickness=2), 'All')
    img1 = image.copy() # copy image to not change it

    # draw contours over image (using original to better see)
    # using the same image in the next two instructions will draw both contours (choosing different colors)
    img1 = cv2.drawContours(img1, #image on which drawing the contours -it will be modified
                            internal_contours, #list of contours to draw
                            contourIdx=-1, #-1 means all the contours, otherwise the index of an element
                            color=(0, 255, 0), #color in BGR
                            thickness=8) #thickness of the line, -1 will fill
    img1 = cv2.drawContours(img1, ratio_contours, contourIdx=-1, color=(150, 0, 0), thickness=2)
    show_img(img1, 'Internal') # my function to show images, just cv2.imshow()/cv2.waitKey()

    #printing the number of cells
    print('internal: ', len(internal_contours), '\t separating or double: ', len(ratio_contours))


if __name__ == '__main__':
    main()
