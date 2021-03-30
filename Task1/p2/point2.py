import cv2
import numpy as np

def padding(img,n):
    print(img)
    R,C = img.shape
    print(R)
    print(C)
    imgAfterPadding = np.zeros((R+n-1,C+n-1))
    imgAfterPadding[1:1+R,1:1+C] = img.copy()
    return(imgAfterPadding)

def imgWithNoise():
        image = cv2.imread(r"D:\CV\Task1\p2\imgWithNoise7.png",0)
        print(image)
        return(padding(image,3))

def AvgFilter():
    img = imgWithNoise()
    R,C = img.shape()
    n = 3 # size of the mask
    mask = np.ones((3,3),np.float32)/9
    newImage = np.zeros((R+n-1,C+n-1))
    for i in range(1,R-2):
        for j in range(1,C-2):
                newImage[i-1,j-1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
    return(newImage)

def medianFilter(img,R,C,n):
    newImage = np.zeros((R+n-1,C+n-1))
    for i in range(1,R-2):
        for j in range(1,C-2):
            mask = img[i:i+n,j:j+n]
            newImage[i-1,j-1] = np.median(mask[:])
    return(newImage)

def normalization(img):
    maxIntensity = np.max(img)
    minIntensity=np.min(img)
    img = (img-minIntensity)/(maxIntensity-minIntensity)
    return(img)

def main():
    img = imgWithNoise()
    R,C = img.shape
    n = 3
    newImage = gaussianFilter(img)
    # newImage = medianFilter(img,R,C,n)
    cv2.imwrite(r"white.png",newImage)
    readWhiteImage=cv2.imread(r"white.png")
    cv2.imshow("After padding", readWhiteImage)
    cv2.waitKey(0)

def gaussianFilter(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img-mean)/std
    return(img) 
    
      
result = main()
