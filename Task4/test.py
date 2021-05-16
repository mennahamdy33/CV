import math 
from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random
from collections import defaultdict
import operator
import Kmeans as Kmeans
import cv2

img = cv2.imread('D:\CV\Task#4\Image-Segmentation-using-K-Means\input2.jpg',1)  

# img = Image.open('D:\CV\Task#4\Image-Segmentation-using-K-Means\input1.jpg')  
# arr = np.asarray(img)

kmean = Kmeans.Kmeans()
output = kmean.Kmeans_Color(img)
print(output)
im_rgb = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_BGR2RGB)

Image.fromarray(im_rgb).save('D:\CV\Task#4\Image-Segmentation-using-K-Means\salma1.jpg')
# Image.fromarray(im_rgb)
plt.figure()
plt.imshow(output)
plt.show()