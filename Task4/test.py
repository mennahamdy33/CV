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
import Ostu

img = cv2.imread(r'D:\CV\CV\Task4\images\Pyramids2.jpg')  

# img = Image.open('D:\CV\Task#4\Image-Segmentation-using-K-Means\input2.jpg')  
 
# arr = np.asarray(img)
kmean = Kmeans.Kmeans()
output =np.array(kmean.Kmeans_Color(img)) 
print(output.shape)

path = "D:\CV\CV\Task4\kmeans.png"
cv2.imwrite(path, output)
N = 4

def LocalThresholding(Image):
        R,C = Image.shape[:2]
        newImg = np.zeros((R,C))
        hR = R // N
        hC = C // N
        

        for i in range(N):
            for j in range(N):
                
                mask = Image[i * hR:hR * (i + 1), j * hC:hC * (j + 1)]
                R,C = mask.shape[:2]
            
                threshold = Ostu.Ostu().Ostu(mask)
                mask[mask < threshold] = 0
                mask[mask > threshold] = 255
                newImg[i * hR:hR * (i + 1), j * hC:hC * (j + 1)] = mask

        path = "D:\CV\CV\Task4\local.png"
        cv2.imwrite(path, newImg)
        # ui.outputTab9.setPixmap(QPixmap(path))
        
# def GlobalThresholding(Image,Th):
#     newImg = Image
#     newImg[newImg < Th] = 0
#     newImg[newImg > Th] = 255
#     path = "D:\CV\CV\Task4\global.png"
#     cv2.imwrite(path, newImg)
#     # ui.outputTab9.setPixmap(QPixmap(path))
# Th = Ostu.Ostu().Ostu(img)
# print(Th)
# GlobalThresholding(img,Th)
img = cv2.imread('D:\CV\Task#4\Otsu-Thresholding\img\org.jpg')
LocalThresholding(img) 

# # Image.fromarray(im_rgb).save('D:\CV\Task#4\Image-Segmentation-using-K-Means\salma1.jpg')
# # # Image.fromarray(im_rgb)
# # plt.figure()
# # plt.imshow(output)


# # # R,C = img.shape
# # # newImg = np.zeros((R,C))
# # # hR = R//2
# # # hC = C//2
# # # for i in range(2):
# # #     for j in range(2):
# # #         mask = img[i*hR:hR*(i+1),j*hC:hC*(j+1)]
# # #         Th = 125
# #         mask[mask<Th] = 0
# #         mask[mask>Th] = 255     
# #         newImg[i*hR:hR*(i+1),j*hC:hC*(j+1)] = mask

# # def func1():return 1
# # def func2():return 2
# # def func3():return 3

# # fl = [func1,func2,func3]

# # print(fl[0]())

# # grey_l = [[40,40,40],[80,80,80],[120,120,120],[160,160,160],[200,200,200],[240,240,240]]
# # print(grey_l[0][5])
