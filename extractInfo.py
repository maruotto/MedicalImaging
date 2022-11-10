import easygui
import unicodedata
import pydicom as pdcm
import cv2
import numpy as np
from image import show_img


def img_path_read():
    uni_code = easygui.fileopenbox(msg="Choose a file to open", default=r"/Users/idamaruotto/Downloads/Mammotest/")
    img_path = unicodedata.normalize('NFKD', uni_code).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')
    return img_path


def main():
    img_path = img_path_read()
    ds = pdcm.dcmread(img_path)
    name = ds.data_element("PatientName")
    print("NAME ", name)
    for e in ds.elements():
        print(e.tag)

    show_img(ds.pixel_array)


def from_zero_to_one(array):
    array = array.astype(np.float32)
    min = np.min(array)
    max = np.max(array)
    return (array - min)/ (max - min)


if __name__ == '__main__':
    main()
