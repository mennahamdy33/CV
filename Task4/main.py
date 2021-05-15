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


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.kselect.hide()
        # self.ui.loadTab1.clicked.connect(lambda: self.getPicrures(1))
       
     
    
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
