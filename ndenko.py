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
# # 964.82
# print(fx)
# # 643
# print(cx)
# # 484
# print(cy)
# # 4 * 4
# print(G_camera_image.shape)
# # 1228800 * 2
# print(LUT.shape)
import numpy as np

import numpy as np

def find_fundamental_matrix(src_points, dst_points):
    if len(src_points) > 7 & len(dst_points) > 7:
        return None

    results = []
    for src_pt, dst_pt in zip(src_points, dst_points):
        x2, y2 = dst_pt
        x1, y1 = src_pt

        results.append([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])

    results = np.array(results)
    # get the smallest eigen value of the right most eigen vector
    _, _, V = np.linalg.svd(results)
    F = np.reshape(V[-1], (3, 3))

    # Rank 2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U,np.diag(S)) @ Vt
    return F



# intrinsic matrix K
# contains info about cameras internal parameters
# focal length fx, and fy,
# principal point cx and cy.
# expressed in image coordinates?
# camera is centered here
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
R_0 = np.eye(3)
t_0 = [[0],[0],[1]]
X_0 = np.hstack((R_0, t_0))
X_0 = np.vstack((X_0, [0,0,0,1]))
print("Initial X matrix")
print(X_0)
# 4 * 4

# 1 0 0 0
# 0 1 0 0
# 0 0 1 1
# 0 0 0 1

X_list = []
X_list.append(X_0)

# Load images from folder
folder_path = './Oxford_dataset_reduced/images/'
image_files = os.listdir(folder_path)

# Initialize ORB detector
orb = cv2.ORB_create()

# store all the good matches 2D arrays we find in this array
good_matches_list = []
# Match keypoints between successive images,
# we do len-1 so we don't go out of bounds
for i in range(len(image_files)-1):
    # Load images and convert to color image
    img1 = cv2.imread(os.path.join(folder_path, image_files[i]))
    img2 = cv2.imread(os.path.join(folder_path, image_files[i+1]))
    # print(type(img1))
    # cv2.imshow('image 1', img1)
    # cv2.waitKey(0)  # wait for any key press
    # cv2.destroyAllWindows()  # close all windows

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # convert color
    # color_image1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    # color_image2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)

    # fix distortion
    # undistorted_image1 = UndistortImage(color_image1, LUT)
    # undistorted_image2 = UndistortImage(color_image2, LUT)

    # Compute keypoints and descriptors
    # kp1, des1 = orb.detectAndCompute(undistorted_image1, None)
    # kp2, des2 = orb.detectAndCompute(undistorted_image2, None)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2,k=2)
    # print("matches:")
    # print(matches)
    print(f'Working on images {i} and {i + 1}')

    # Apply ratio test to remove ambiguous matches
    good_matches = []
    good_matches1D = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            good_matches1D.append(m)
    print("Number of good matches")
    print(len(good_matches))
    print("Good Matches: ")
    print(good_matches)

    # we need at least 8 corresponding points to do this process
    if len(good_matches) >= 8:

        # find fundamental matrix for those 2 images
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches1D]).reshape(-1, 1, 2)
        src = np.float32([kp1[m.queryIdx].pt for m in good_matches1D])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches1D]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good_matches1D])

        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 0.1, 0.99)
        F_test = find_fundamental_matrix(src,dst)
        print("Source points: ")
        print(src_pts)
        print("Fundamental Matrix is ")
        print(F)
        print("F shape is: ")
        print(np.shape(F))
        print("F test is: ")
        print(F_test)
        # Even with 8 we can still fail to find F
        if np.shape(F) == (3,3):
            #  Estimate the Essential Matrix E from the Fundamental Matrix F by accounting for the
            # calibration parameters.
            E = K.T @ F @ K
            print("Essential Matrix is: ")
            print(E)

            # decompose E into translation T and rotation R
            _, R, T, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=K)


            # print("retval")
            # print(retval)
            print("R")
            print(R)
            print("T")
            print(T)

            X_k = np.hstack((R,T))
            X_k = np.vstack((X_k, [0,0,0,1]))
            print("X_k is currently: ")
            print(X_k)
            # multiply X_k with 1st item that was X_list, then append it
            # to the front of the list
            # X_list.insert(0,(X_k @ X_list[0]))

            # multiply X_k with last item that was X_list, then append it
            # to the end of the list
            X_list.append(X_k @ X_list[-1])


# takes an index and a list of matrixes and
# multiplies up until that index
def multiply_through(index, list):
    if index == 0:
        return list[0]

    curr = list[0]
    i = 1
    while i <= index:
        curr = curr @ list[i]
        i += 1
    return curr


print("X_list: ")
print(X_list)
#Now that we have our list,the product of every element takes in
#a point in camera 1 coord and spits out camera K's
# We dont want that, we want input camera K coord and spitout 1's

# invert every matrix in X_list
# for matrix in X_list:
#     matrix = np.linalg.inv(matrix)

# invert every matrix in our list
invertedX_list = [np.linalg.inv(matrix) for matrix in X_list]
original_center = np.array([[0], [0], [0], [1]])

results = []
i = 1


# while i < len(invertedX_list):
#     result = multiply_through(i, invertedX_list)
#     result = result @ original_center
#     results.append(result)
#     i += 1

while i < len(invertedX_list):
    result = invertedX_list[i] @ original_center
    results.append(result)
    i += 1

# while i < len(X_list):
#     result = X_list[i] @ original_center
#     results.append(result)
#     i += 1



print("Results: ")
print(results)

# goes through a list of vectors and plots the x and y of each
def plot_points2D(results):
    # get all the x values from the vectors
    x = []
    i = 0
    while i < len(results):
        x.append(results[i][0])
        i += 1

    print("x: ")
    print(x)
    # get all the y values from the vectors
    y = []
    i = 0
    while i < len(results):
        y.append(results[i][2])
        i += 1

    print("y: ")
    print(y)
    # plot them
    plt.plot(x,y)
    # plt.scatter(x,y)
    # Set the labels for x and y axes
    plt.xlabel('x')
    plt.ylabel('y')

    # Set the title of the plot
    plt.title('Trajectory')

    # Show the plot
    plt.show()

plot_points2D(results)

