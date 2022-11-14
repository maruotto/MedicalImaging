import easygui
from image import normalize, show_img
import unicodedata
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename

#import numpy as np
import cv2

def img_path_read():
    tk().withdraw()
    img_path = askopenfilename(message="Choose a file to open", initialdir=r"/Users/idamaruotto/Downloads/Segmentaziones/dataset/train/")
    print(img_path)
    #uni_code = easygui.fileopenbox(msg="Choose a file to open",
    #                               default=r"/Users/idamaruotto/Downloads/Segmentaziones/dataset/train/")
    img_path = unicodedata.normalize('NFKD', img_path).encode('ascii', 'ignore')
    img_path = img_path.decode('utf-8')
    #img_path = "/Users/idamaruotto/Downloads/Segmentaziones/dataset/train/00001_p0_patch0.tif"
    return img_path

def main():
    img_path = img_path_read()
    src = cv2.imread(img_path,0)
    #src = normalize(src)
    print("coso")
    #cv2.imshow('name', src)
    show_img(src)
    cv2.waitKey()

    #dst = cv2.LUT(src.astype(np.uint8), lut)
    #show_img(dst, 'Gamma corrected image')


if __name__ == '__main__':
    main()
