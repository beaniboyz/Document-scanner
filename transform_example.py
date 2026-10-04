from transform import four_point_transform
import numpy as np
import argparse
import cv2

#Construct the argument parse and parse the argument to the dictionary with "vars"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image file")
ap.add_argument("-c", "--coords", required=True, help="list of source points")
args = vars(ap.parse_args)

#load the image and grab the source points
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

warped = four_point_transform(image, pts)

#SHow the original and warped images
cv2.imshow("Orginal", image)
cv2.imshow("Warped", warped)

#Keep two window on and free after turning off
cv2.waitKey(0)
cv2.destroyAllWindows()