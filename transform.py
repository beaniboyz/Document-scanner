import numpy as np
import cv2

def order_points(pts):
    #Initialize a list of coordinates with 0( 4 row, 2 column)
    #The order rectangle is top-left -> top-right -> bottom-right -> bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    #The top-left will have the smallest sum, whereas the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    #The top-right will have the smallest difference, the bottom-left will have the largest difference
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect