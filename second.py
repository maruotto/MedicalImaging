import easygui
import unicodedata

import numpy as np
import pydicom as pdcm
import cv2
from image import show_img, normalize, gamma_transform, show_no_wait_img


def img_path_read():
    uni_code = easygui.fileopenbox(msg="Choose a file to open", default=r"/Users/idamaruotto/Downloads/Mammotest/")
    img_path = unicodedata.normalize('NFKD', uni_code).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')
    return img_path


def main():
    img_path = img_path_read()
    ds = pdcm.dcmread(img_path)
    pixel_array = ds.pixel_array
    src = normalize(pixel_array)
    show_no_wait_img(src, 'Original Image')
    lut = gamma_transform(0.5)

    dst = cv2.LUT(src.astype(np.uint8), lut)
    show_img(dst, 'Gamma corrected image')


if __name__ == '__main__':
    main()
