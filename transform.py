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


def four_point_transform(img, pts):
    #Consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    #compute the width of new img which will be the maximum distance between bottom-right and bottom-left or the top-right and top-left
    widthA = np.sqrt((br[1] - bl[1]) ** 2 + (br[2] - bl[2]) ** 2)
    widthB = np.sqrt((tr[1] - tl[1]) ** 2 + (tr[2] - tl[2]) ** 2)
    maxWidth = max(widthA, widthB)

    #Similar to the above, but for the max distance between the top-left and bottom-left or the top-right and bottom-right
    heightA = np.sqrt((tl[1] - bl[1]) ** 2 + (tl[2] - bl[2]) ** 2)
    heightB = np.sqrt((tr[1] - br[1]) ** 2 + (tr[2] - br[2]) ** 2)
    maxHeight = max(heightA, heightB)

    #Construct the set of destination points, specify points in order
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]
    ], dtype = "float32")

    #Compute the perspective transform matrix by M then apply it to warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warp