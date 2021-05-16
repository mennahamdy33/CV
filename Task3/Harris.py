import numpy as np
import cv2
import glob
from scipy.ndimage.filters import convolve

import time


class Harris:

    def __init__(self):
        pass

    def all(self,Image1):
        image = cv2.imread(Image1)
        greyimg = self.rgb2gray(image)
        w, h = greyimg.shape
        R = self.HarrisCornerDetection(greyimg)
        CornerStrengthThreshold = 10000
        radius = 3
        color = (0, 255, 0)  # Green
        thickness = 1
        PointList = []

        for row in range(w):
            for col in range(h):
                if R[row][col] > CornerStrengthThreshold:
                    max = R[row][col]
                    skip = False
                    for nrow in range(5):
                        for ncol in range(5):
                            if row + nrow - 2 < w and col + ncol - 2 < h:
                                if R[row + nrow - 2][col + ncol - 2] > max:
                                    skip = True
                                    break

                    if not skip:
                        cv2.circle(image, (col, row), radius, color, thickness)
                        PointList.append((row, col))
        return image,PointList

    def HarrisCornerDetection(self,image):
        w, h = image.shape
        ImgX = cv2.Sobel(image, cv2.CV_64F,1 , 0, ksize=1)
        ImgY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

        for ind1 in range(w):
            for ind2 in range(h):
                if ImgY[ind1][ind2] < 0:
                    ImgY[ind1][ind2] *= -1
                if ImgX[ind1][ind2] < 0:
                    ImgX[ind1][ind2] *= -1

        ImgX_2 = np.square(ImgX)
        ImgY_2 = np.square(ImgY)

        ImgXY = np.multiply(ImgX, ImgY)
        ImgYX = np.multiply(ImgY, ImgX)


        ImgX_2 = cv2.GaussianBlur(ImgX_2,(5,5),0)
        ImgY_2 = cv2.GaussianBlur(ImgY_2,(5,5),0)
        ImgXY = cv2.GaussianBlur(ImgXY,(5,5),0)
        ImgYX = cv2.GaussianBlur(ImgYX,(5,5),0)

        alpha = 0.06
        R = np.zeros((w, h), np.float32)

        for row in range(w):
            for col in range(h):
                M_bar = np.array([[ImgX_2[row][col], ImgXY[row][col]], [ImgYX[row][col], ImgY_2[row][col]]])
                R[row][col] = np.linalg.det(M_bar) - (alpha * np.square(np.trace(M_bar)))
        return R

    def rgb2gray(self,rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
