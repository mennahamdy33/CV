import math 
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import scipy as sp
import random
from collections import defaultdict
import operator
import Kmeans as Kmeans
import cv2
import numpy as np

img = cv2.imread('D:\CV\Task#4\Image-Segmentation-using-K-Means\input2.jpg',1)  

img = Image.open('D:\CV\Task#4\Image-Segmentation-using-K-Means\input2.jpg')  
 
arr = np.asarray(img)

kmean = Kmeans.Kmeans()
output = kmean.Kmeans_Color(arr)

output1 = kmean.convertColorIntoGray(output) 
# print(output)
# im_rgb = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_BGR2RGB)

# Image.fromarray(im_rgb).save('D:\CV\Task#4\Image-Segmentation-using-K-Means\salma1.jpg')
# # Image.fromarray(im_rgb)
plt.figure()
plt.imshow(output)

plt.figure()
plt.imshow(output1)
plt.show()
# R,C = img.shape
# newImg = np.zeros((R,C))
# hR = R//2
# hC = C//2
# for i in range(2):
#     for j in range(2):
#         mask = img[i*hR:hR*(i+1),j*hC:hC*(j+1)]
#         Th = 125
#         mask[mask<Th] = 0
#         mask[mask>Th] = 255     
#         newImg[i*hR:hR*(i+1),j*hC:hC*(j+1)] = mask
# cv2.imwrite("D:\CV\Task#4\Otsu-Thresholding\img\LocalThresholding.png", newImg)

# def func1():return 1
# def func2():return 2
# def func3():return 3

# fl = [func1,func2,func3]

# print(fl[0]())
