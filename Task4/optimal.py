import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import isfile , join
from PIL import Image
from skimage.transform import resize
import cv2

class Optimal:

    def __init__(self):
        pass

    def optimal_thresholding(self,thImg):

        bg_sum = (thImg[0, 0] + thImg[0, -1] + thImg[-1, 0] + thImg[-1, -1])
        fg_sum = np.sum(thImg) - bg_sum
        bg_mean = bg_sum / 4

        fg_mean = fg_sum / (np.size(thImg) - 4)

        t = (bg_mean + fg_mean) / 2
        while True:
            bg_mean = np.mean(thImg[thImg < t])
            fg_mean = np.mean(thImg[thImg >= t])
            if t == (bg_mean + fg_mean) / 2:
                break
            t = (bg_mean + fg_mean) / 2
        return t

