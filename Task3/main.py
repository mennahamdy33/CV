from PyQt5 import QtWidgets, QtGui,QtCore
from imageview import Ui_MainWindow
import sys
import numpy as np
import cv2
import sift
import time
import importlib
from PIL import Image
import Harris
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.n = 3
        self.precent = 0.1
        self.alpha = 0.8
        self.minIntensity = 0
        self.maxIntensity = 255
        self.Th = 150
        self.original_RGB = None
        self.original_gray = None
        self.outputTabs = [self.ui.output1Tab8,
                            self.ui.output2Tab8]
        self.Template = None
        self.ui.menuExit.triggered.connect(exit)
        self.ui.loadTab1.clicked.connect(lambda: self.getPicrures(1))
        self.ui.loadTab2.clicked.connect(lambda: self.getPicrures(2))
        self.ui.load1Tab3.clicked.connect(lambda: self.getPicrures(3))
        self.ui.load2tab3.clicked.connect(lambda: self.getPicrures(4))
        self.ui.loadEdgeImg.clicked.connect(lambda: self.getPicrures(5))
        self.ui.loadTab6.clicked.connect(lambda: self.getPicrures(6))
        self.ui.load1Tab8.clicked.connect(lambda: self.getPicrures(7))
        self.ui.load2Tab8.clicked.connect(lambda: self.getPicrures(8))
        self.ui.applyTab6.clicked.connect(self.HarrisDetect)
        self.ui.SIFTTab8.clicked.connect(self.SIFT)
        self.ui.correlationTab8.clicked.connect(lambda:self.TemplateMatching(1))
        self.ui.ssdTab8.clicked.connect(lambda:self.TemplateMatching(0))
     
    def HarrisDetect(self):
        start = time.time()
        self.Harris = Harris.Harris()
        img,keypoints = self.Harris.all(self.HarrisPath)
        cv2.imwrite("./images/Harris.png",img)
        w = self.ui.outputTab6.width()
        h = self.ui.outputTab6.height()
        self.ui.outputTab6.setPixmap(QPixmap("./images/Harris.png").scaled(w,h,QtCore.Qt.KeepAspectRatio))
        end = time.time()
        timeNeeded = (end - start )/60
        timeNeeded = round(timeNeeded,2)
        self.ui.timeTab6.setText(str(timeNeeded))

    def SIFT(self):
        start = time.time()
        self.sift = sift.Sift(self.path1,self.path2)
        done = self.sift.sift_timing()
        end = time.time()
        timeNeeded = (end - start )/60 
        timeNeeded = round(timeNeeded,2)
        self.ui.timeTab8.setText(str(timeNeeded))

    def TemplateMatching(self,flag):
        start = time.time()
        self.output = self.sift.OutPut(flag)
        end = time.time()
        w = self.ui.output1Tab8.width()
        h = self.ui.output2Tab8.height()
        self.outputTabs[flag].setPixmap(QPixmap("./images/M"+str(flag)+".png").scaled(w,h,QtCore.Qt.KeepAspectRatio))
        timeNeeded = (end - start )/60 
        timeNeeded = round(timeNeeded,2)
        self.ui.timeMatchTab8.setText(str(timeNeeded))

    def getPicrures(self, tab):
        path, extention = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
 
            if (tab == 1):
                self.ui.inputTab1.setPixmap(QPixmap(path))
            elif (tab == 2):
                self.ui.inputTab2.setPixmap(QPixmap(path))
            elif (tab == 3):
                self.ui.input1Tab3.setPixmap(QPixmap(path))
                self.LowCompImage = self.padded
            elif (tab == 4):
                self.ui.input2Tab3.setPixmap(QPixmap(path))
                self.HighCompImage = self.padded
            elif (tab == 5):
                self.ui.InputTab4.setPixmap(QPixmap(path))
            elif (tab == 6):
                w = self.ui.inputTab6.width()
                h = self.ui.inputTab6.height()
                self.ui.inputTab6.setPixmap(QPixmap(path).scaled(w,h,QtCore.Qt.KeepAspectRatio))
                self.HarrisPath = path    
            elif (tab == 7):
                w = self.ui.input1Tab8.width()
                h = self.ui.input1Tab8.height()
                self.ui.input1Tab8.setPixmap(QPixmap(path).scaled(w,h,QtCore.Qt.KeepAspectRatio))
                self.path1 = path
            elif (tab == 8):
                w = self.ui.input2Tab8.width()
                h = self.ui.input2Tab8.height()
                self.ui.input2Tab8.setPixmap(QPixmap(path).scaled(w,h,QtCore.Qt.KeepAspectRatio))
                self.path2 = path  
           
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
