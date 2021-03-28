from PyQt5 import QtWidgets,QtGui
from ImageViewer import Ui_MainWindow
import sys
import numpy as np
import pandas as pd
import cv2
QPixmap = QtGui.QPixmap


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.padded =[]
        self.R=0
        self.C=0
        self.n=3
        self.precent=0.1
        self.ui.actionExit.triggered.connect(exit)
        self.ui.actionInput.triggered.connect(self.getPicrures)
        self.ui.noise.clicked.connect(lambda: self.salt_pepper_noise(self.image,self.precent))
        self.ui.filter.clicked.connect(lambda:self.AvgFilter(self.img_noisy,self.R,self.C,self.n))
        self.ui.edge.activated.connect(self.chooseEdge)

    def rgb2gray(self, rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    def salt_pepper_noise(self,img, percent):
        print(percent)
        self.img_noisy = np.zeros(img.shape)
        print(self.R,self.C)
        salt_pepper = np.random.random(img.shape)  # Uniform distribution
        cleanPixels_ind = salt_pepper > percent
        NoisePixels_ind = salt_pepper <= percent
        pepper = (salt_pepper <= (0.5 * percent))  # pepper < half percent
        salt = ((salt_pepper <= percent) & (salt_pepper > 0.5 * percent));
        self.img_noisy[cleanPixels_ind] = img[cleanPixels_ind]
        self.img_noisy[pepper] = 0
        self.img_noisy[salt] = 1
        cv2.imwrite(r"./images/noise.png", self.img_noisy)
        self.ui.output1.setPixmap(QPixmap(r"./images/noise.png"))

    def chooseEdge(self):
        if str(self.ui.edge.currentText())=="Sobel":
            self.SobelFilter(self.newImage,self.R,self.C,self.n)
        if str(self.ui.edge.currentText())=="Prewitt ":
            self.PrewittFilter(self.newImage,self.R,self.C,self.n)
        if str(self.ui.edge.currentText())=="Roberts" :   
            self.RobertsFilter(self.newImage,self.R,self.C,2)

    def AvgFilter(self,img,R,C,n):
        mask = np.ones((3,3),np.float32)/9
        self.newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    self.newImage[i+1,j+1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
        cv2.imwrite("Filtered.png",self.newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Filtered.png"))

    def SobelFilter(self,img,R,C,n):
        maskX = [[-1,0,1],[-2,0,2],[-1,0,1]]
        maskY = [[1,2,1],[0,0,0],[-1,-2,-1]]  
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                S1 = np.sum(np.sum(np.multiply(maskX,img[i:i+n,j:j+n])))
                S2 = np.sum(np.sum(np.multiply(maskY,img[i:i+n,j:j+n])))
                newImage[i+1,j+1]= np.sqrt(np.sqrt(np.power(S1,2)+np.power(S2,2)))
                if newImage[i+1,j+1] > 12:
                    newImage[i+1,j+1] = 200
        cv2.imwrite("Edge.png",newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png")) 
    def PrewittFilter(self,img,R,C,n):
        maskX = [[-1,0,1],[-1,0,1],[-1,0,1]]
        maskY = [[1,1,1],[0,0,0],[-1,-1,-1]] 
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    S1 = np.sum(np.multiply(maskX,img[i:i+n,j:j+n]))
                    S2 = np.sum(np.multiply(maskY,img[i:i+n,j:j+n]))
                    newImage[i+1,j+1] = np.sqrt(np.sqrt(np.power(S1,2)+np.power(S2,2)))
                    if newImage[i+1,j+1] > 12:
                        newImage[i+1,j+1] = 200
        cv2.imwrite("Edge.png",newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png"))
    def RobertsFilter(self,img,R,C,n):
        maskX = [[1,0],[0,-1]]
        maskY = [[0,1],[-1,0]]
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    S1 = np.sum(np.multiply(maskX,img[i:i+n,j:j+n]))
                    S2 = np.sum(np.multiply(maskY,img[i:i+n,j:j+n]))
                    newImage[i+1,j+1] = np.sqrt(np.sqrt(np.power(S1,2)+np.power(S2,2)))
                    # if newImage[i+1,j+1] > 10:
                    #     newImage[i+1,j+1] = 200
        cv2.imwrite("Edge.png",newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png"))   
         
    def padding(self,img,n):
        self.R,self.C= img.shape
        imgAfterPadding = np.zeros((self.R+self.n-1,self.C+self.n-1))
        imgAfterPadding[1:1+self.R,1:1+self.C] = img.copy()
        return(imgAfterPadding)       
        
    def getPicrures (self):
        path,extention = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            self.image = cv2.imread(path,0)
            self.padded = self.padding(self.image,self.n)
            self.ui.Input1.setPixmap(QPixmap(path))

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()        