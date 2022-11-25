
from image import show_img, show_no_wait_img, img_path_read
from binary_img_lib import contour_bounding_box, calculate_area_thresh, contour_of_not_borders, ratio_borders
import cv2

MINIMUM_AREA = 500


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
