import unicodedata
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename
from image import show_img, show_no_wait_img
import cv2
import numpy as np


def img_path_read():
    tk().withdraw()
    img_path = askopenfilename(message="Choose a file to open", initialdir=r"/Users/idamaruotto/Downloads/Mammotest")
    print(img_path)
    # uni_code = easygui.fileopenbox(msg="Choose a file to open",
    #                               default=r"/Users/idamaruotto/Downloads/Mammotest")
    img_path = unicodedata.normalize('NFKD', img_path).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')
    # img_path = "/Users/idamaruotto/Downloads/Segmentaziones/dataset/train/00001_p0_patch0.tif"
    return img_path


def evaluate_seed(dim, mask):
    zeros = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if (i==0 or i==dim-1) or (j==0 or j==dim-1):
                if(mask[i,j]==255):
                    zeros[i, j] = 255
    return zeros


def execute(m1, kernel, mask):
    m0 = m1.copy()
    m1 = cv2.dilate(m0, kernel)
    m1 = np.where(m1 == mask,255,0)
    return m1.astype(np.uint8)


def main():
    global image
    img_path = "/Users/idamaruotto/Downloads/Mammotest/acanthocyte1.jpg"
    image = cv2.imread(img_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # transform to grey level - single channel
    _, img = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # opening to remove small elements

    # create kernel
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # Using cv2.erode() method eroded = cv2.erode(image, kernel)
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #show_no_wait_img(img, 'Before')
    #show_no_wait_img(opening, 'Without small elements')

    seed = evaluate_seed(img.shape[0], img)
    #print(mask)

    # reconstruct by dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    m0 = img.copy()
    m1 = execute(seed, kernel, img)
    print(m1.dtype)
    i = 0
    while not np.array_equiv(m0, m1):
        m0 = m1.copy()
        m1 = execute(m1, kernel, img)
        show_img(m1, 'After')
        if i == 10000:
            exit(4)
        i += 1
    # Displaying the image
    print(i)
    show_no_wait_img(m1, 'After')


if __name__ == '__main__':
    main()
    cv2.waitKey()
