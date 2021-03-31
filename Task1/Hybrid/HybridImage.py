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
        self.R=0
        self.C=0
        self.n=3
        self.flag =0 
        self.alpha = 0.9
        self.precent=0.1
        self.ui.actionExit.triggered.connect(exit)
        self.ui.actionInput.triggered.connect(self.getPicrures)
        self.ui.hybrid.clicked.connect(self.Hybrid)

    def gaussianFilter(self,img):
        self.R , self.C = img.shape
        print(img.shape)
        mean = np.mean(img)
        std = np.std(img)
        img = (img-mean)/std
        cv2.imwrite("Filtered.png",img)
        return(img)    
    def AvgFilter(self,img,R,C,n):
        mask = np.ones((3,3),np.float32)/9
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    newImage[i+1,j+1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
        cv2.imwrite("Filtered.png",newImage)            
        return newImage            
        # self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Filtered.png"))

    def LaplacianFilter(self,img,R,C,n):

        mask = [[0,-1,0],[-1,4,-1],[0,-1,0]]
          
        newImage = np.zeros((R,C))
        for i in range(1,R-2):
            for j in range(1,C-2):
                newImage[i+1,j+1] = np.sum(np.sum(np.multiply(mask,img[i:i+n,j:j+n])))
                
        # newImage *= 255.0 / newImage.max()
        cv2.imwrite("Edge.png",newImage)
        return newImage
        
        # self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png"))     
    def Hybrid(self):
        # imageSmoothed = self.AvgFilter(self.image,self.R,self.C,self.n)
        imageSmoothed = self.gaussianFilter(self.image)
        imageSharped = self.LaplacianFilter(self.image2,self.R,self.C,self.n)
        print(imageSmoothed.shape)
        print(imageSharped.shape)
        outputImage = (imageSmoothed * (1-self.alpha)) + (imageSharped*self.alpha)
        # outputImage = imageSmoothed + imageSharped
        cv2.imwrite("Hybride.png",outputImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Hybrid/Hybride.png"))

    def padding(self,img,n):
        R,C= img.shape
        imgAfterPadding = np.zeros((R+n-1,C+n-1))
        imgAfterPadding[1:1+R,1:1+C] = img.copy()
        return(imgAfterPadding)     
    
    def getPicrures (self):
        path,extention = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:

            self.flag += 1
            if self.flag == 1 :
                self.image = cv2.imread(path,0)
                print(self.image.shape)
                self.padded1 = self.padding(self.image,5)
                print(self.padded1.shape)
                self.ui.Input1.setPixmap(QPixmap(path))
            elif self.flag ==2 :
                self.image2 = cv2.imread(path,0)
                print(self.image2.shape)
                self.padded2 = self.padding(self.image2,self.n)
                print(self.padded2.shape)
                self.ui.input2.setPixmap(QPixmap(path))
                self.flag =0 
            else :
                pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()        