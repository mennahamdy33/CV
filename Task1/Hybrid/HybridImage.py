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
        self.flag =0 
        self.precent=0.1
        self.ui.actionExit.triggered.connect(exit)
        self.ui.actionInput.triggered.connect(self.getPicrures)
        self.ui.hybrid.clicked.connect(self.Hybrid)
    def AvgFilter(self,img,R,C,n):
        mask = np.ones((3,3),np.float32)/9
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    newImage[i+1,j+1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
        cv2.imwrite("Filtered.png",newImage)            
        return newImage            
        # self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Filtered.png"))

    def SobelFilter(self,img,R,C,n):
        maskX = [[-1,0,1],[-2,0,2],[-1,0,1]]
        maskY = [[1,2,1],[0,0,0],[-1,-2,-1]]  
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                S1 = np.sum(np.sum(np.multiply(maskX,img[i:i+n,j:j+n])))
                S2 = np.sum(np.sum(np.multiply(maskY,img[i:i+n,j:j+n])))
                newImage[i+1,j+1]= np.sqrt(np.power(S1,2)+np.power(S2,2))
                
        newImage *= 255.0 / newImage.max()
        cv2.imwrite("Edge.png",newImage)
        return newImage
        
        # self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png"))     
    def Hybrid(self):
        imageSmoothed = self.AvgFilter(self.image,self.R,self.C,self.n)
        imageSharped = self.SobelFilter(self.image2,self.R,self.C,self.n)
        outputImage = imageSmoothed * (1.0 - 0.2) + imageSharped * 0.2
        # outputImage = imageSmoothed + imageSharped
        cv2.imwrite("Hybride.png",outputImage)
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

            self.flag += 1
            if self.flag == 1 :
                self.image = cv2.imread(path,0)
                self.padded = self.padding(self.image,self.n)
                self.ui.Input1.setPixmap(QPixmap(path))
            elif self.flag ==2 :
                self.image2 = cv2.imread(path,0)
                self.padded = self.padding(self.image2,self.n)
                self.ui.input2.setPixmap(QPixmap(path))
            else :
                pass    


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()        