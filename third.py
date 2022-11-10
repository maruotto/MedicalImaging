import easygui
import unicodedata

import numpy as np
import pydicom as pdcm
import cv2
from image import normalize, show_no_wait_img, two_histograms
import matplotlib.pyplot as plt

clip_limit_slider_max = 10
tile_size_slider_max = 100

src = None
sub = None
fig = None


def on_trackbar(val):
    global src
    global sub
    global fig
    clahe = cv2.createCLAHE(clipLimit=val)
    cl1 = clahe.apply(src).astype(np.uint8)
    show_no_wait_img(cl1, 'CLAHE')

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
    clahe = cv2.createCLAHE()
    cl1 = clahe.apply(src).astype(np.uint8)
    show_no_wait_img(src, 'Original Image')
    show_no_wait_img(cl1, 'CLAHE')

    fig, sub = two_histograms(src, cl1, "Original Image Histogram", "CLAHE modified Image Histogram")

    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)
    clip_limit_trackbar_name = 'Clip limit: '
    cv2.createTrackbar(clip_limit_trackbar_name, "Trackbars", 0, clip_limit_slider_max, on_trackbar)
    tile_size_trackbar_name = 'Tile size: '
    cv2.createTrackbar(tile_size_trackbar_name, "Trackbars", 1, tile_size_slider_max, on_trackbar)


if __name__ == '__main__':
    main()
    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
