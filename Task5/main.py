from PyQt5 import QtWidgets, QtGui,QtCore
from ImageView import Ui_MainWindow
import sys
import numpy as np
import cv2
import time
import importlib
import matplotlib.gridspec as gridspec
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from PIL import Image
from pylab import *
from skimage.transform import resize
import DetectionFunction

import math

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.loadTab1.clicked.connect(lambda: self.getPicrures(1))
        self.ui.faceDetection.clicked.connect(self.FaceDetection)

    def FaceDetection(self):
        image = cv2.imread(self.Path)
        image_copy = image.copy()
        haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
        haar_cascade_SideFace = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

        gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        
        faces_rect = haar_cascade_face.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 2)
        print("Faces Founds:" , len(faces_rect))
        if (len(faces_rect) != 0):
            for (x,y,w,h) in faces_rect:
                cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,0,0),2)
        
        elif (len(faces_rect)==0):
            SideFace = haar_cascade_SideFace.detectMultiScale(gray_image,scaleFactor = 1.1, minNeighbors = 1)
            print("Side faces Founds:" , len(SideFace))
            for (sx,sy,sw,sh) in SideFace:
                cv2.rectangle(image_copy,(sx,sy),(sx+sw,sy+sh),(255,255,255),2)
                
        cv2.imwrite("Detection_Output.jpg",image_copy)        
        w = self.ui.outputTab1.width()
        h = self.ui.outputTab1.height()
        self.ui.outputTab1.setPixmap(QPixmap("Detection_Output.jpg").scaled(w, h, QtCore.Qt.KeepAspectRatio))        

    def getPicrures(self, tab):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")

        if path == "":
            pass
        else:
            if tab == 1:
                w = self.ui.inputTab1.width()
                h = self.ui.inputTab1.height()
                self.ui.inputTab1.setPixmap(QPixmap(path).scaled(w, h, QtCore.Qt.KeepAspectRatio))
                self.Path = path 
            
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
