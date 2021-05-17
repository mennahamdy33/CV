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

    # firstimage = cv2.imread("./images/Pyramids2.jpg")
    # imgs2_preprocessed = segmentation_resize(np.array(firstimage))
    #
    #
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(imgs2_preprocessed.astype('float32'), cv2.COLOR_BGR2RGB))
    # plt.savefig('old.png', dpi=300, bbox_inches='tight')
    # plt.clf()
    # if len(imgs2_preprocessed.shape) == 3: imgs2_preprocessed = rgb2gray(imgs2_preprocessed)
    # t = optimal_thresholding(imgs2_preprocessed)
    # b = plt
    # b.hist(imgs2_preprocessed.flatten(), 256)
    # b.axvline(x=t, color='r', linestyle='dashed', linewidth=2)
    # b.savefig('hist.png', dpi=300, bbox_inches='tight')
    # b.clf()
    # # plt.show()
    # c = plt
    #
    # c.axis('off')
    # c.imshow(imgs2_preprocessed >= t)
    # c.savefig('new.png', dpi=300, bbox_inches='tight')