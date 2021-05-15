import numpy as np
from scipy import ndimage
import math
from random import randint
import matplotlib.pyplot as plt
import cv2

class meanShift:
    H = 90
    Hr = 90
    Hs = 90
    Iter = 100
    Mode = 2
    
    def __init__(self , Image1):
        self.img = cv2.imread(Image1,cv2.IMREAD_COLOR)
        self.opImg = np.zeros(self.img.shape,np.uint8)
        self.boundaryImg = np.zeros(self.img.shape,np.uint8)

    def output(self):  
        self.performMeanShift(self.img)
        return(self.opImg)

    def getNeighbors(self,seed,matrix,mode=2):
        neighbors = []
        nAppend = neighbors.append
        sqrt = math.sqrt
        for i in range(0,len(matrix)):
            cPixel = matrix[i]
            # if mode is 1, we threshold using H
            if (mode == 1):
                d = sqrt(sum((cPixel-seed)**2))
                if(d<self.H):
                    nAppend(i)
            # otherwise, we threshold using H
            else:
                r = sqrt(sum((cPixel[:3]-seed[:3])**2))
                s = sqrt(sum((cPixel[3:5]-seed[3:5])**2))
                if(s < self.Hs and r < self.Hr ):
                    nAppend(i)
        return neighbors

    def markPixels(self,neighbors,mean,matrix,cluster):
        for i in neighbors:
            cPixel = matrix[i]
            x=cPixel[3]
            y=cPixel[4]
            self.opImg[x][y] = np.array(mean[:3],np.uint8)
            self.boundaryImg[x][y] = cluster
        return np.delete(matrix,neighbors,axis=0)    

    def calculateMean(self,neighbors,matrix):
        neighbors = matrix[neighbors]
        r=neighbors[:,:1]
        g=neighbors[:,1:2]
        b=neighbors[:,2:3]
        x=neighbors[:,3:4]
        y=neighbors[:,4:5]
        mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])
        return mean  

    def createFeatureMatrix(self,img):
        h,w,d = img.shape
        F = []
        FAppend = F.append
        for row in range(0,h):
            for col in range(0,w):
                r,g,b = img[row][col]
                # l , u , v = LUVConvertion(r,g,b)
                FAppend([r,g,b,row,col])
                # FAppend([l,u,v,row,col])
        F = np.array(F)
        return F

    def performMeanShift(self,img):
        clusters = 0
        F = self.createFeatureMatrix(img)
        while(len(F) > 0):
            randomIndex = randint(0,len(F)-1)
            seed = F[randomIndex]
            initialMean = seed
            neighbors = self.getNeighbors(seed,F,self.Mode)
            if(len(neighbors) == 1):
                F=self.markPixels([randomIndex],initialMean,F,clusters)
                clusters+=1
                continue
            mean = self.calculateMean(neighbors,F)
            meanShift = abs(mean-initialMean)
            if(np.mean(meanShift)<self.Iter):
                F = self.markPixels(neighbors,mean,F,clusters)
                clusters+=1
        return clusters