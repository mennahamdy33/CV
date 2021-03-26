import cv2
import numpy as np

def padding(img,n):
    R,C = img.shape
    imgAfterPadding = np.zeros((R+n-1,C+n-1))
    imgAfterPadding[1:1+R,1:1+C] = img.copy()
    return(imgAfterPadding)

def imgWithNoise():
        image = cv2.imread(r"D:\CV\Task1\p2\imgWithNoise7.png",0)
        return(padding(image,3))

def AvgFilter():
    img = imgWithNoise()
    R,C = img.shape
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

 

def main():
    img = imgWithNoise()
    R,C = img.shape
    n = 3
    newImage = medianFilter(img,R,C,n)
    cv2.imwrite(r"white.png",newImage)
    readWhiteImage=cv2.imread(r"white.png")
    cv2.imshow("After padding", readWhiteImage)
    cv2.waitKey(0)
    
    
    
    
    

    

    
    
result = main()