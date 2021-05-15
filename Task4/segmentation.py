import numpy as np
from scipy import ndimage
import math
from random import randint
import matplotlib.pyplot as plt
import cv2

class meanShift:
    Hr = 90
    Hs = 90
    Iter = 100

    def __init__(self , Image1):
        self.img = cv2.imread(Image1,cv2.IMREAD_COLOR)
        # self.img = cv2.cvtColor(self.img,cv2.COLOR_RGB2LUV)
        self.outputImg = np.zeros(self.img.shape)

    def output(self):  
        self.meanShift(self.img)
        # self.opImg = cv2.cvtColor(self.opImg,cv2.COLOR_LUV2RGB)
        return(self.outputImg)

    def neighbors(self,randomFeatures,features):
        neighbors = []
        for i in range(0,len(features)):
            pixel = features[i]
            r = math.sqrt(sum((pixel[:3]-randomFeatures[:3])**2))
            s = math.sqrt(sum((pixel[3:5]-randomFeatures[3:5])**2))
            if(s < self.Hs and r < self.Hr ):
                neighbors.append(i)
        return neighbors

    def markPixels(self,neighbors,mean,features):
        for i in neighbors:
            pixel = features[i]
            x=pixel[3]
            y=pixel[4]
            self.outputImg[x][y] = np.array(mean[:3])
        return np.delete(features,neighbors,axis=0)    

    def mean(self,neighbors,features):
        neighbors = features[neighbors]
        r=neighbors[:,:1]
        g=neighbors[:,1:2]
        b=neighbors[:,2:3]
        x=neighbors[:,3:4]
        y=neighbors[:,4:5]
        mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])
        return mean  

    def features(self,img):
        r,c,_ = img.shape
        Features = []
        for row in range(0,r):
            for col in range(0,c):
                r,g,b = img[row][col]
                # l , u , v = LUVConvertion(r,g,b)
                Features.append([r,g,b,row,col])
                # FAppend([l,u,v,row,col])
        Features = np.array(Features)
        return Features

    def meanShift(self,img):
        clusters = 0
        feature = self.features(img)
        while(len(feature) > 0):
            index = randint(0,len(feature)-1)
            randomFeature = feature[index]
            initialCluster = randomFeature
            neighbors = self.neighbors(randomFeature,feature)
            if(len(neighbors) == 1):
                feature=self.markPixels([index],initialCluster,feature)
                clusters+=1
                continue
            mean = self.mean(neighbors,feature)
            meanShift = abs(mean-initialCluster)
            if(np.mean(meanShift)<self.Iter):
                feature = self.markPixels(neighbors,mean,feature)
                clusters+=1
        return clusters