from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import argparse
import imutils

#STEP 1: EDGE DETECTION
#Construct the argument parser and parse  the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args)

#load the image
img = cv2.imread("image")
org = img.copy()

#Find ratio of the old height to the new height and resize it to 500 pixel in order to scan fast and convenient
ratio = img.shape[0] / 500.0
img = imutils.resize(img, height=500)

#Convert the image to grayscale, blur it, and find edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#it's kernel 5 x 5, 0 means tell openCV automatically calculate based on Kernel size
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#below 75 will be discarded, exceed 200 will be retained
edged = cv2.Canny(gray, 75, 200)

#Keep two window on,then free memory after turning off
print("Step 1: Edge detection")
cv2.imshow("Image", img)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()