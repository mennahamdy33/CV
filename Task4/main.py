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
        self.ui.meanshiftTab10.clicked.connect(self.meanshift)
        self.ui.kmeansTab10.clicked.connect(self.Kmeans)
        self.Path = ""
        self.Image = None
        self.grayImage = None
        self.outputTabs = [self.ui.output1Tab10,self.ui.output2Tab10]

    def chooseOptimalThreshold(self):
        if str(self.ui.optimalTab9.currentText()) == "Global Thresholding":

            if len(self.thImg.shape) == 3: self.thImg = self.rgb2gray(self.thImg)
            a = optimal.Optimal()
            threshold = a.optimal_thresholding(self.thImg)
            plt.hist(self.thImg.flatten(), 256)
            plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
            plt.savefig('./images/optimalHistogram.png', dpi=300, bbox_inches='tight')
            plt.clf()
            # self.ui..setPixmap(QPixmap("./images/optimalHistogram.png"))
            plt.figure(figsize=(2, 2))
            plt.axis('off')
            plt.imshow(self.thImg >= threshold)

            plt.savefig('./images/optimalImage.png', dpi=300, bbox_inches='tight')
            plt.clf()
            self.ui.outputTab9.setPixmap(QPixmap("./images/optimalImage.png"))
        if str(self.ui.optimalTab9.currentText()) == "Local Thresholding":
            # if len(self.thImg.shape) == 3: self.thImg = self.rgb2gray(self.thImg)
            # n = 100
            # small_area = n*n
            # big_area = self.thImg.shape[0]*self.thImg.shape[1]
            # noOfBlocks = big_area/small_area
            # newImg = np.zeros((self.thImg.shape[0], self.thImg.shape[1]))
            # # fig = plt.figure(figsize=(2, 2))
            # fig, ax = plt.subplots(int(noOfBlocks/2),int(noOfBlocks/2), figsize=(3, 3))
            # for r in range(0,self.thImg.shape[0] - n, n):
            #     for c in range(0, self.thImg.shape[1] - n, n):
            #         window = self.thImg[r:r + n, c:c + n]
            #         a = optimal.Optimal()
            #         threshold = a.optimal_thresholding(window)
            #         #         plt.hist(self.thImg.flatten(), 256)
            #         #         plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
            #         # plt.savefig('./images/optimalHistogram.png', dpi=300, bbox_inches='tight')
            #         # plt.clf()
            #         # self.ui..setPixmap(QPixmap("./images/optimalHistogram.png"))
            #         subplot = ax[r, c]
            #         subplot.axis('off')
            #         subplot.imshow(window >= threshold)
            #
            #         plt.savefig('./images/optimalImage'+str(r+c)+'.png', dpi=300)
            #         plt.clf()
            # self.ui.outputTab9.setPixmap(QPixmap("./images/optimalImage.png"))

            # small_area = n*n
            # big_area = self.thImg.shape[0]*self.thImg.shape[1]
            # noOfBlocks = big_area/small_area
            # print(noOfBlocks)
            #fig, ax = plt.subplots((math.ceil(self.thImg.shape[0]/n)),(math.ceil(self.thImg.shape[1]/100)), figsize=(3, 3))
            n =10
            if len(self.thImg.shape) == 3: self.thImg = self.rgb2gray(self.thImg)
            plt.figure(figsize=(3, 3))
            gs1 = gridspec.GridSpec((math.ceil(self.thImg.shape[0]/n)), (math.ceil(self.thImg.shape[1]/n)))
            gs1.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.

            print((math.ceil(self.thImg.shape[1]/n)))
            print((math.ceil(self.thImg.shape[0] / n)))

            self.R = self.thImg.shape[0]
            self.C = self.thImg.shape[1]
            i=0

            for x in range(self.R):
                for y in range(self.C):
                    a = optimal.Optimal()
                    if y%n==0 and x%n==0:
                        print(x, y)
                        threshold = a.optimal_thresholding(self.thImg[x:x + n, y:y + n])

                        subplot = plt.subplot(gs1[i])
                        subplot.axis('off')
                        subplot.imshow(self.thImg[x:x + n, y:y + n] >= threshold )
                        i = i+1
            #
            # plt.subplots_adjust( wspace=0.0,
            #         hspace=0.0)
            plt.savefig('./images/optimalLocalImage.png', dpi=300)

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
    #
    # def rgb2gray(self,path):
    #     self.Image = cv2.imread(path,1)
    #     self.grayImage = cv2.imread(path,0)
    #     # cv2.imwrite("D:\CV\CV\Task4\images\grayImage.png",self.grayImage)
    #     # self.ui.input2Tab10.setPixmap(QPixmap("D:\CV\CV\Task4\images\grayImage.png"))

    def rgb2gray(self , rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

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
            if (tab == 0):

                img = cv2.imread(path)
                self.thImg = np.float32(self.segmentation_resize(img)) *255
                cv2.imwrite("./images/resizedImage.png",self.thImg)
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
