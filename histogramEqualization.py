import easygui
import unicodedata

import numpy as np
import pydicom as pdcm
import cv2
from image import normalize, show_no_wait_img, two_histograms
import matplotlib.pyplot as plt

src = None
sub = None
fig = None


def on_trackbar(val):
    global src
    global sub
    global fig
    cl1 = cv2.equalizeHist(src).astype(np.uint8)
    show_no_wait_img(cl1, 'Hist Equalized')
    sub[1].clear()
    sub[1].hist(cl1.ravel(), np.max(cl1), [0, np.max(cl1)])
    plt.draw()
    fig.canvas.flush_events()


def img_path_read():
    uni_code = easygui.fileopenbox(msg="Choose a file to open", default=r"/Users/idamaruotto/Downloads/Mammotest/")
    img_path = unicodedata.normalize('NFKD', uni_code).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')
    return img_path


def main():
    global src
    global sub
    global fig
    img_path = img_path_read()
    ds = pdcm.dcmread(img_path)
    pixel_array = ds.pixel_array
    src = normalize(pixel_array)
    cl1 = cv2.equalizeHist(src).astype(np.uint8)
    show_no_wait_img(src, 'Original Image')
    show_no_wait_img(cl1, 'HE')

    fig, sub = two_histograms(src, cl1, "Original Image Histogram", "HE modified Image Histogram")


if __name__ == '__main__':
    main()
    cv2.waitKey()
    plt.show()
    cv2.destroyAllWindows()
