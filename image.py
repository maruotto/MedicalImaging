import numpy as np
import cv2
import math
#import easygui
import unicodedata
from matplotlib import pyplot as plt


def img_path_read():
    uni_code = easygui.fileopenbox(msg="Choose a file to open", default=r"/Users/idamaruotto/Downloads/Mammotest/")
    img_path = unicodedata.normalize('NFKD', uni_code).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')
    return img_path


# ##################### IMAGE SHOW ############################
def show_img(array, name='Image'):
    cv2.namedWindow(name)
    cv2.imshow(name, array)
    # cv2.resizeWindow(name, 800, 800)
    cv2.waitKey()
    cv2.destroyWindow(name)


def show_no_wait_img(array, name='Image'):
    cv2.namedWindow(name)
    cv2.imshow(name, array)
    # cv2.resizeWindow(name, 800, 800)


# ##################### NORMALIZATION ############################
def from_zero_to_one(array):
    array = array.astype(np.float32)
    minimum = np.min(array)
    maximum = np.max(array)
    return (array - minimum) / (maximum - minimum)


def normalize_image(pixel_array, max_value=255):
    tmp = pixel_array.astype(np.float32)
    return tmp/max_value


def normalize(array, bit=8):
    tmp = from_zero_to_one(array) * (math.pow(2, bit)-1)
    tmp = tmp.astype(np.uint8)
    return tmp

# ##################### GAMMA TRANSFORM ############################


def gamma_transform_v2(gamma, max_value=255):
    lut = [0]
    for i in range(1, max_value+1):
        k = float(i)
        k = k/max_value
        k = math.pow(k, gamma)
        k = k * max_value
        lut += [k]
    lut = np.array(lut, dtype=np.uint8)
    print(lut)
    return np.clip(lut, 0, max_value)


def gamma_transform(gamma, max_value=255):
    lut = [0]
    for i in range(1, max_value+1):
        k = float(i)
        c = math.pow(max_value+1, 1-gamma)
        k = math.pow(k, gamma)
        lut += [c*k]
    lut = np.array(lut, dtype=np.uint8)
    return np.clip(lut, 0, max_value)

# ##################### HISTOGRAMS ############################


def sub_histogram(subfigure, image, title=""):
    """create an histogram of the image in the figure
        It uses matplotlib
        It requires a next call to matplotlib.pyplot.show() to produce effects"""
    subfigure.hist(image.ravel(), np.max(image), [0, np.max(image)])
    subfigure.set_title(title)


def histogram(image):
    """create an histogram of the image in a new figure
        It uses matplotlib
        It requires a next call to matplotlib.pyplot.show() to produce effects"""
    plt.hist(image.ravel(), np.max(image), [0, np.max(image)])


def two_histograms(im1, im2, first_title="", second_title="", window_title="Histograms", shy=False, shx=False):
    fig, axes = plt.subplots(2, 1, sharex=shx, sharey=shy)
    fig.canvas.manager.set_window_title(window_title)
    sub_histogram(axes[0], im1, first_title)
    sub_histogram(axes[1], im2, second_title)
    fig.tight_layout()
    return fig, axes
