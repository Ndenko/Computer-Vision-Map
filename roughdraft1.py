import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from ReadCameraModel import ReadCameraModel


# cv2.imshow('Rotated image1',rotated_image1)

# _____________3.1________________________________
# this finds the variables of a cameras intrinsic matrix
from UndistortImage import UndistortImage

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./Oxford_dataset_reduced/model')
# 964.82
print(fx)
# 643
print(cx)
# 484
print(cy)
# 4 * 4
print(G_camera_image.shape)
# 1228800 * 2
print(LUT.shape)


# intrinsic matrix K
# contains info about cameras internal parameters
# focal length fx, and fy,
# principal point cx and cy.
# expressed in image coordinates?
# camera is centered here
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

# _____________3.2________________________________

# get the image (black and white by default) then
# convert it to color
img1 = cv2.imread("./Oxford_dataset_reduced/images/1399381446267196.png",0)
img2 = cv2.imread("./Oxford_dataset_reduced/images/1399381446892097.png",flags=-1)
imgX = cv2.imread("./Oxford_dataset_reduced/images/1399381573374966.png",0)

color_image1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
color_image2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
color_imageX = cv2.cvtColor(imgX, cv2.COLOR_BayerGR2BGR)



# undistort it
undistorted_image1 = UndistortImage(color_image1,LUT)
undistorted_image2 = UndistortImage(color_image1,LUT)
undistorted_imageX = UndistortImage(color_imageX,LUT)

# cv2.imshow('Color Image',undistorted_image1)
# cv2.imshow('Color Image',undistorted_image2)

print(type(img1))
print(type(undistorted_image1))
# _____________3.3________________________________



# Initialize ORB detector and descriptor extractor
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(undistorted_image1, None)
kp2, des2 = orb.detectAndCompute(undistorted_image2, None)

# Initialize brute-force matcher
bf = cv2.BFMatcher()

# Match keypoints and filter matches
matches = bf.knnMatch(des1, des2,k=2)
print(len(matches))
# matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))
good_matches = []
for m,n in matches:
    # print(matches[0].distance)
    # print(m.distance)
    # print(0.75 * matches[0].distance)
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
print(good_matches)
print(type(good_matches))

print(kp1)
print(kp2)
# Draw correspondences and display image
img_matches = cv2.drawMatchesKnn(undistorted_image1, kp1, undistorted_image2, kp2,good_matches,None, flags=2)

# cv2.imshow('Color Image1',undistorted_image1)
# cv2.waitKey(1000)
# cv2.imshow('Color Image2',undistorted_imageX)
cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matches', 600, 400)
cv2.imshow('Matches', img_matches)




cv2.waitKey(0) # wait for any key press
cv2.destroyAllWindows() # close all windows



