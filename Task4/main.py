from PyQt5 import QtWidgets, QtGui,QtCore
from ImageView import Ui_MainWindow
import sys
import numpy as np
import cv2
import time
import importlib
from PIL import Image
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import segmentation
import Kmeans as Kmeans
from PIL import Image
from pylab import *

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.kselect.hide()
        self.ui.load1Tab10.clicked.connect(lambda: self.getPicrures(1))
        self.ui.meanshiftTab10.clicked.connect(self.meanshift)
        self.ui.kmeansTab10.clicked.connect(self.Kmeans)
        self.Path = ""
        self.Image = None
        self.grayImage = None
        self.outputTabs = [self.ui.output1Tab10,self.ui.output2Tab10]

    def meanshift(self):    
        self.mean = segmentation.meanShift(self.Path)
        imgOutput = self.mean.output()
        cv2.imwrite("./images/meanshiftMethod.png",imgOutput)
        w = self.ui.output1Tab10.width()
        h = self.ui.output1Tab10.height()
        self.ui.output1Tab10.setPixmap(QPixmap("./images/meanshiftMethod.png").scaled(w,h,QtCore.Qt.KeepAspectRatio))
        self.mean2 = segmentation.meanShift("./images/grayImage.png")
        imgOutput2 = self.mean2.output()
        cv2.imwrite("./images/GraymeanshiftMethod.png",imgOutput2)
        w = self.ui.output2Tab10.width()
        h = self.ui.output2Tab10.height()
        self.ui.output2Tab10.setPixmap(QPixmap("./images/GraymeanshiftMethod.png").scaled(w,h,QtCore.Qt.KeepAspectRatio))

    def rgb2gray(self,path):
        self.Image = cv2.imread(path,1)
        self.grayImage = cv2.imread(path,0)
        cv2.imwrite("D:\CV\CV\Task4\images\grayImage.png",self.grayImage)
        self.ui.input2Tab10.setPixmap(QPixmap("D:\CV\CV\Task4\images\grayImage.png"))
     

    def getPicrures(self, tab):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            if (tab == 1):
                w = self.ui.input1Tab10.width()
                h = self.ui.input1Tab10.height()
                self.ui.input1Tab10.setPixmap(QPixmap(path))
                self.rgb2gray(path)
                self.Path = path 
    
    def Kmeans(self):
        kmeans = Kmeans.Kmeans()
       
        RGBKmeansOutput = kmeans.Kmeans_Color(self.Image)
        grayKmeansOutput = kmeans.convertColorIntoGray(RGBKmeansOutput)
       
        self.showOutput([RGBKmeansOutput,grayKmeansOutput])

    def showOutput(self,outputData):
        for i in range(2):
            path = "D:\CV\CV\Task4\images\output"+str(i)+".png"
            cv2.imwrite(path,outputData[i])
            w = self.outputTabs[i].width()
            h = self.outputTabs[i].height()
            self.outputTabs[i].setPixmap(QPixmap(path).scaled(w,h,QtCore.Qt.KeepAspectRatio))
    
    def LocalThresholding(self,Th):
            n = 5 
            newImg = np.zeros((self.R,self.C))   
            for i in range(self.R):
                for j in range(self.C):
                    mask = np.mean(self.img[i:i+n,j:j+n]) 
                    newImg[i,j] = 255 if mask >= Th else 0
            cv2.imwrite("D:\CV\Task#4\Otsu-Thresholding\img\LocalThresholding.png", newImg)
        
    def GlobalThresholding(self,Th):
        newImg = np.zeros(( self.R, self.C))
        for i in range( self.R):
            for j in range( self.C):
                newImg[i,j] = 255 if self.img[i,j] >= Th else 0
        cv2.imwrite("D:\CV\Task#4\Otsu-Thresholding\img\GlobalThresholding.png", newImg)
           


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
