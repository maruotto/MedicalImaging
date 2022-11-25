import cv2

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
