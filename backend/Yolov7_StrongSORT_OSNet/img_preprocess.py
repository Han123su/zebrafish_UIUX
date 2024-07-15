import cv2
import scipy
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt


def get_data(path):
    im = cv2.imread(str(path))  # , cv2.IMREAD_GRAYSCALE
    return im


def nothing(x):
    pass


def img_preprocessing(img):
    winName = 'Colors of the rainbow'
    # cv2.namedWindow(winName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = img[:, :, 1]
    blur_img = cv2.GaussianBlur(gray, (7, 7), 4)  # not img[:,:,0]
    # ret3, binary = cv2.threshold(g1, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33,
                                   6)  # 35, 8 31 6

    # cv2.namedWindow('binary', cv2.WINDOW_AUTOSIZE)

    binary = binary > 0
    binary = scipy.ndimage.binary_fill_holes(binary)
    # binary = morphology.remove_small_objects(binary, min_size=1000)

    binary = np.uint8(binary * 255)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    # binary = binary > 0
    # binary = morphology.remove_small_objects(binary, min_size=500)
    binary = scipy.ndimage.binary_fill_holes(binary)
    binary = np.uint8(binary * 255)
    binary = cv2.dilate(binary, kernel, iterations=1)
    h, w = binary.shape
    binary[0:2, :] = 0
    binary[:, 0:2] = 0
    binary[h - 3:h, :] = 0
    binary[:, w - 3:w] = 0
    # cv2.imshow('binary', binary)
    # cv2.waitKey(0)
    binary = binary > 0
    return binary



