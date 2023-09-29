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
# image_files = os.listdir(folder_path)



# store all the good matches 2D arrays we find in this array
good_matches_list = []
# Retrieve and sort the image file paths
# image_paths = sorted(glob.glob(folder_path))


imgs = []
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img,cv2.COLOR_BayerGR2BGR)
        img = UndistortImage(img, LUT)
        if img is not None:
            imgs.append(img)
        print(len(imgs))

print("done")
# Initialize ORB detector

bf = cv2. BFMatcher()
r_base = np.eye (3)
t_base = [[0], [0], [1]]
X_0 = np.hstack((r_base, t_base))
X_0 = np.vstack((X_0, [0,0,0,1]))
X_list = []
X_list.append(X_0)
x_coord = [0]
y_coord = [0]
z_coord = [0]
sift = cv2.xfeatures2d.SIFT_create()
for i in range (len(imgs)-1):
    keypoints1, descriptors1 = sift.detectAndCompute(imgs[i], None)
    keypoints2, descriptors2 = sift.detectAndCompute(imgs[i+1], None)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    dest = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    F, mask = cv2. findFundamentalMat (src, dest, cv2. FM_RANSAC)

    E = K.T.dot(F).dot(K)
    points, R, T, mask = cv2.recoverPose(E, src, dest, K)
    X_k = np.hstack((R, T))
    X_k = np. vstack((X_k, [0,0,0,1]))

    # multiply X_k with last item that was X_list, then append it
    # to the end of the list
    X_list.append(X_k @ X_list[-1])
    print(X_k)




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

# invert every matrix in our list
invertedX_list = [np.linalg.inv(matrix) for matrix in X_list]
original_center = np.array([[0], [0], [0], [1]])
#
results = []
i = 1

while i < len(invertedX_list):
    result = invertedX_list[i] @ original_center
    results.append(result)
    i += 1

plt.xlabel('x')
plt.ylabel('y')

# Set the title of the plot
plt.title('Trajectory')
plt.plot(x_coord, z_coord)

# Show the plot
plt.show()

# print("Results: ")
# print(results)

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
    z = []
    i = 0
    while i < len(results):
        z.append(results[i][2])
        i += 1

    print("z: ")
    print(z)
    # plot them


    plt.plot(x, z)
    # plt.scatter(x_coord, z_coord)
    # plt.plot(x,y)
    # plt.scatter(x,y)

    # Set the labels for x and y axes
    plt.xlabel('x')
    plt.ylabel('z')

    # Set the title of the plot
    plt.title('Trajectory')

    # Show the plot
    plt.show()

plot_points2D(results)