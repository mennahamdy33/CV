from PyQt5 import QtWidgets, QtGui,QtCore
from ImageView import Ui_MainWindow
import sys
import numpy as np
import cv2
import time
import segmentation
import importlib
from PIL import Image
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.kselect.hide()
        self.ui.load1Tab10.clicked.connect(lambda: self.getPicrures(1))
        self.ui.meanshiftTab10.clicked.connect(self.meanshift)
        self.Path = ""

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
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        grayImage = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        cv2.imwrite("./images/grayImage.png",grayImage)
        w = self.ui.input2Tab10.width()
        h = self.ui.input2Tab10.height()
        self.ui.input2Tab10.setPixmap(QPixmap("./images/grayImage.png"))
     
    
    def getPicrures(self, tab):
        path, extention = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            if (tab == 1):
                w = self.ui.input1Tab10.width()
                h = self.ui.input1Tab10.height()
                self.ui.input1Tab10.setPixmap(QPixmap(path))
                self.rgb2gray(path)
                self.Path = path 
           
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
