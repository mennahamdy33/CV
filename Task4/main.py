from PyQt5 import QtWidgets, QtGui,QtCore
from ImageView import Ui_MainWindow
import sys
import numpy as np
import cv2
import time
import importlib
import matplotlib.gridspec as gridspec
from PIL import Image
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import segmentation
import Kmeans as Kmeans
from PIL import Image
from pylab import *
from skimage.transform import resize
import optimal
import math
import Ostu
# import agglo_segmentation

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.kselect.hide()
        self.ui.load1Tab10.clicked.connect(lambda: self.getPicrures(1))
        self.ui.loadTab9.clicked.connect(lambda: self.getPicrures(0))
        self.ui.optimalTab9.activated.connect(self.chooseOptimalThreshold)
        self.ui.otsuTab9.activated.connect(self.ostu)
        self.ui.meanshiftTab10.clicked.connect(self.meanshift)
        self.ui.kmeansTab10.clicked.connect(self.Kmeans)
        self.ui.agglomerativeTab10.clicked.connect(self.agglomerative)

        self.Path = ""
        self.Image = None
        self.grayImage = None
        self.outputTabs = [self.ui.output1Tab10,self.ui.output2Tab10]
        self.Th_functions = [optimal.Optimal().optimal_thresholding,
                            Ostu.Ostu().Ostu]
        self.N = 8 #Threshold Size
        

    def chooseOptimalThreshold(self):
        status = str(self.ui.optimalTab9.currentText())
        if status == "Global Thresholding":
            Th = self.Th_functions[0](self.thImg)
            self.GlobalThresholding(self.thImg,Th)
        else:
            self.LocalThresholding(self.thImg,0)

        
          
    def segmentation_resize(self,img):
        ratio = min(1, np.sqrt((512 * 512) / np.prod(img.shape[:2])))
        newshape = list(map(lambda d: int(round(d * ratio)), img.shape[:2]))
        img = resize(img, newshape, anti_aliasing=True)
        return img

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

    def rgb2gray(self , rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

    def getPicrures(self, tab):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            if tab == 1:
                self.Image = cv2.imread(path)
    
                self.ui.input1Tab10.setPixmap(QPixmap(path))
                rgbImage = cv2.imread(path,cv2.IMREAD_COLOR)
                cv2.imwrite(".\images\grayImage.png",self.rgb2gray(rgbImage))
                w1 = self.ui.input2Tab10.width()
                h1 = self.ui.input2Tab10.height()
                self.ui.input2Tab10.setPixmap(QPixmap(".\images\grayImage.png").scaled(w1,h1,QtCore.Qt.KeepAspectRatio))
                self.Path = path 
            if tab == 0:

                img = cv2.imread(path)
                self.grayThImage = cv2.imread(path,0)
                self.thImg = np.float32(self.segmentation_resize(img)) *255
            
                cv2.imwrite("./images/resizedImage.png",self.thImg)
                if len(self.thImg.shape) == 3: self.thImg = self.rgb2gray(self.thImg)

                self.ui.inputTab9.setPixmap(QPixmap("./images/resizedImage.png"))
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
    
    def LocalThresholding(self, Image, flag):
            R,C = Image.shape[:2]
            newImg = np.zeros((R,C))
            hR = R // self.N
            hC = C // self.N

            for i in range(self.N):
                for j in range(self.N):
                    mask = Image[i * hR:hR * (i + 1), j * hC:hC * (j + 1)]

                    threshold = self.Th_functions[flag](mask)
                    mask[mask < threshold] = 0
                    mask[mask > threshold] = 255
                    newImg[i * hR:hR * (i + 1), j * hC:hC * (j + 1)] = mask
    
            path = "./images/LocalOutPut.png"
            cv2.imwrite(path, newImg)
            self.ui.outputTab9.setPixmap(QPixmap(path))
            
    def GlobalThresholding(self,Image,Th):
        newImg = Image
        newImg[newImg < Th] = 0
        newImg[newImg > Th] = 255
        path = "./images/GlobalOutPut.png"
        cv2.imwrite(path, newImg)
        self.ui.outputTab9.setPixmap(QPixmap(path))
        

    def agglomerative(self):
        # use the mask to select the "interesting" part of the image
        # sel = np.zeros_like(image)
        # sel[mask] = image[mask]

        n_clusters = 3
        img = cv2.imread(self.Path, cv2.IMREAD_UNCHANGED)

        pixels = img.reshape((-1, 3))
        agglo = agglo_segmentation.AgglomerativeClustering(pixels, k=n_clusters, initial_k=25, )
        agglo.fit(pixels)
        new_img = [[agglo.predict_center(list(pixel)) for pixel in row] for row in img]
        new_img = np.array(new_img, np.uint8)
        grayAggloImg = self.rgb2gray(new_img)
        cv2.imwrite("./images/aggloMethod.png", new_img)
        cv2.imwrite("./images/grayAggloMethod.png", grayAggloImg)
        self.ui.output1Tab10.setPixmap(QPixmap("./images/aggloMethod.png"))
        self.ui.output2Tab10.setPixmap(QPixmap("./images/grayAggloMethod.png"))

    def ostu(self):
        status = str(self.ui.otsuTab9.currentText())
        if (status == 'Global Thresholding'):
            Th = self.Th_functions[1](self.grayThImage)
            self.GlobalThresholding(self.grayThImage,Th)
        else:
            self.LocalThresholding(self.thImg, 1)

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
