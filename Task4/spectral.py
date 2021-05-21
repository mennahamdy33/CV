import numpy as np
import cv2 
import math

def spectral_threshold(image):
    blur = cv2.GaussianBlur(image,(5,5),0)
    hist = cv2.calcHist([image],[0],None,[256],[0,256]) 
    hist /= float(np.sum(hist)) 
    BetweenClassVarsList = []
    for bar, _ in enumerate(hist):
        Foregroundlevels = []
        BackgroundLevels = []
        ForegroundHist = []
        BackgroundHist = []
        for level, value in enumerate(hist):
            if level < bar:
                BackgroundLevels.append(level)
                BackgroundHist.append(value)
            else:
                Foregroundlevels.append(level)
                ForegroundHist.append(value)
        
        FCumSum = np.sum(ForegroundHist)
        BCumSum = np.sum(BackgroundHist)
        FMean = np.sum(np.multiply(ForegroundHist, Foregroundlevels)) / float(FCumSum)
        BMean = np.sum(np.multiply(BackgroundHist, BackgroundLevels)) / float(BCumSum)  
        GMean = np.sum(np.multiply(hist, range(0, 256)))
        BetClsVar = FCumSum*np.square(FMean - GMean) + BCumSum*np.square(BMean - GMean)
        # print(BetClsVar)
        BetweenClassVarsList.append(BetClsVar)
    # print(max(BetweenClassVarsList))
    return BetweenClassVarsList.index(np.nanmax(BetweenClassVarsList))
image cv2.imread()
spectral_threshold()





