import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
from os import listdir
from os.path import isfile, join


def featureMatch(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    print("good,len", len(good))

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    plt.imshow(img3), plt.show()


# fetch training data
training_path = 'data/master'
only_files = [f for f in listdir(training_path) if isfile(join(training_path, f))]

for image in only_files:
    # extract full image path
    image_path = join(training_path, image)
    print("image_path", image_path)

    mat = cv2.imread(image_path)

    # convert image to HSV
    hsv = cv2.cvtColor(mat, cv2.COLOR_BGR2HSV)

    ## mask of green (36,0,0) ~ (70, 255,255)
    # lower_green = np.array([12, 30, 0])
    # upper_green = np.array([30, 50, 100])
    '''
    color = 25
    sensitivity = 15
    lower_green = np.array([color - sensitivity, 25, 0])
    upper_green = np.array([color + sensitivity, 100, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    '''
    color = 25
    sensitivity = 15
    lower_green = np.array([color - sensitivity, 25, 0])
    upper_green = np.array([color + sensitivity, 100, 180])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    ## slice the green
    imask = mask > 0
    green = np.zeros_like(mat, np.uint8)
    green[imask] = mat[imask]

    # k mean skleanr
    '''
    # reshape the image to be a list of pixels
    reshaped_hsv = green.reshape((mat.shape[0] * mat.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(init='k-means++', n_clusters=10, n_init=10)
    clt.fit(reshaped_hsv)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)
    bar = utils.plot_colors(hist, clt.cluster_centers_)

    cv2.imshow("bar", bar)
    '''

    Z = green.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print("label", label)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((green.shape))

    cv2.imshow('res2', res2)



    other = cv2.add(mat, green, None, mask);
    concatGreen = np.concatenate((mat, other), axis=1)
    # con  cat = np.concatenate((concatGreen, hsv), axis=1)
    cv2.namedWindow("green", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("green", 1000, 600)
    cv2.imshow("green", concatGreen)
    cv2.waitKey(0)

    '''
    histr = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    plt.title("histogram")
    plt.plot(histr, label=image)
    '''

cv2.destroyAllWindows()
'''
plt.legend()
plt.xlim([0, 256])
plt.show()
'''
'''
# reshape the image to be a list of pixels
reshaped_hsv = hsv.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(init='k-means++', n_clusters=20, n_init=10)
clt.fit(reshaped_hsv)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

cv2.imshow("bar", bar)


# concatenate image to display
concat = np.concatenate((image, hsv), axis=1)
cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
cv2.resizeWindow("hsv", 1000,600)
cv2.imshow("hsv", concat)


cv2.waitKey(0)
cv2.destroyAllWindows()
'''
