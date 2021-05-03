from PyQt5 import QtWidgets, QtGui,QtCore
from imageview import Ui_MainWindow
import sys
import numpy as np
import cv2
import sift
import time
# import Point1.cannyEdge

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
        # self.ui.groupBox_2.hide()
        # self.ui.selectTab1.activated.connect(self.chooseFilter)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.loadTab1.clicked.connect(lambda: self.getPicrures(1))
        self.ui.loadTab2.clicked.connect(lambda: self.getPicrures(2))
        self.ui.load1Tab3.clicked.connect(lambda: self.getPicrures(3))
        self.ui.load2tab3.clicked.connect(lambda: self.getPicrures(4))
        self.ui.loadEdgeImg.clicked.connect(lambda: self.getPicrures(5))
        self.ui.Load1Tab7.clicked.connect(lambda: self.getPicrures(7))
        self.ui.Load2Tab7.clicked.connect(lambda: self.getPicrures(8))
        self.ui.Match.clicked.connect(self.SIFT)

        # self.ui.hybrid1.clicked.connect(self.Hybrid)
        # self.ui.set.clicked.connect(self.Normalization)
        # self.ui.gray.clicked.connect(lambda: self.getHistogram(self.grayImg, 'grey'))
        # self.ui.color.clicked.connect(lambda: self.getHistogram(self.image, ' '))
        # self.ui.cumcolor.clicked.connect(lambda: self.getHistogram(self.image, 'c'))
        #self.ui.CannyEdgeBtn.clicked.connect(self.cannyFilter)

    def SIFT(self):
        start = time.time()
        self.sift = sift.Sift(self.path1,self.path2)
        output = self.sift.OutPut()
        cv2.imwrite("./images/siftPicture.png",  cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        w = self.ui.OutputTab7.width()
        h = self.ui.OutputTab7.height()
        self.ui.OutputTab7.setPixmap(QPixmap("./images/siftPicture.png").scaled(w,h,QtCore.Qt.KeepAspectRatio))
        end = time.time()
        timeNeeded = (end - start )/60 
        timeNeeded = round(timeNeeded,2)
        self.ui.TimeTab7.setText(str(timeNeeded))


    def getPicrures(self, tab):
        path, extention = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            # self.image = cv2.imread(path)
            # self.grayImg = self.rgb2gray(self.image)
            # cv2.imwrite(r"./images/grayPicture.png", self.grayImg)
            # self.padded = self.padding(self.grayImg, self.n)
            if (tab == 1):
                self.ui.inputTab1.setPixmap(QPixmap(path))
            elif (tab == 2):
                self.ui.inputTab2.setPixmap(QPixmap(path))
                # test image


            elif (tab == 3):
                self.ui.input1Tab3.setPixmap(QPixmap(path))
                self.LowCompImage = self.padded
            elif (tab == 4):
                self.ui.input2Tab3.setPixmap(QPixmap(path))
                self.HighCompImage = self.padded
            elif (tab == 5):
                self.ui.InputTab4.setPixmap(QPixmap(path))
            elif (tab == 7):
                w = self.ui.Input1Tab7.width()
                h = self.ui.Input1Tab7.height()
                self.ui.Input1Tab7.setPixmap(QPixmap(path).scaled(w,h,QtCore.Qt.KeepAspectRatio))
                self.path1 = path
            elif (tab == 8):
                w = self.ui.Input2Tab7.width()
                h = self.ui.Input2Tab7.height()
                self.ui.Input2Tab7.setPixmap(QPixmap(path).scaled(w,h,QtCore.Qt.KeepAspectRatio))
                self.path2 = path      


    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
