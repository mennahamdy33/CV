from PyQt5 import QtWidgets,QtGui
from imageViewer import Ui_MainWindow
import sys
import numpy as np
import pandas as pd
import cv2
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.padded =[]
        self.R=0
        self.C=0
        self.n=3
        self.precent=0.1
        self.ui.actionExit.triggered.connect(exit)
        self.ui.actionInput.triggered.connect(self.getPicrures)
        self.ui.noise.clicked.connect(lambda: self.salt_pepper_noise(self.image,self.precent))
        self.ui.filter.clicked.connect(lambda:self.AvgFilter(self.img_noisy,self.R,self.C,self.n))
        self.ui.edge.activated.connect(self.chooseEdge)
        self.ui.histogram.clicked.connect(lambda: self.getHistogram(self.image, 'c'))
        self.ui.freqDominFilter.clicked.connect(lambda: self.freqFilter(self.grayImg))

    def rgb2gray(self, rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    def salt_pepper_noise(self,img, percent):
        print(percent)
        self.img_noisy = np.zeros(img.shape)
        print(self.R,self.C)
        salt_pepper = np.random.random(img.shape)  # Uniform distribution
        cleanPixels_ind = salt_pepper > percent
        NoisePixels_ind = salt_pepper <= percent
        pepper = (salt_pepper <= (0.5 * percent))  # pepper < half percent
        salt = ((salt_pepper <= percent) & (salt_pepper > 0.5 * percent))
        self.img_noisy[cleanPixels_ind] = img[cleanPixels_ind]
        self.img_noisy[pepper] = 0
        self.img_noisy[salt] = 1
        cv2.imwrite(r"./images/noise.png", self.img_noisy)
        self.ui.output1.setPixmap(QPixmap(r"./images/noise.png"))

    def chooseEdge(self):
        if str(self.ui.edge.currentText())=="Sobel":
            self.SobelFilter(self.newImage,self.R,self.C,self.n)
        if str(self.ui.edge.currentText())=="Prewitt ":
            self.PrewittFilter(self.newImage,self.R,self.C,self.n)
        if str(self.ui.edge.currentText())=="Roberts" :   
            self.RobertsFilter(self.newImage,self.R,self.C,2)

    def AvgFilter(self,img,R,C,n):
        mask = np.ones((3,3),np.float32)/9
        self.newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    self.newImage[i+1,j+1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
        cv2.imwrite("Filtered.png",self.newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Filtered.png"))

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
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png")) 
    def PrewittFilter(self,img,R,C,n):
        maskX = [[-1,0,1],[-1,0,1],[-1,0,1]]
        maskY = [[1,1,1],[0,0,0],[-1,-1,-1]] 
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    S1 = np.sum(np.multiply(maskX,img[i:i+n,j:j+n]))
                    S2 = np.sum(np.multiply(maskY,img[i:i+n,j:j+n]))
                    newImage[i+1,j+1] = np.sqrt(np.power(S1,2)+np.power(S2,2))
        newImage *= 255.0 / newImage.max()                
        cv2.imwrite("Edge.png",newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png"))
    def RobertsFilter(self,img,R,C,n):
        maskX = [[1,0],[0,-1]]
        maskY = [[0,1],[-1,0]]
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    S1 = np.sum(np.multiply(maskX,img[i:i+n,j:j+n]))
                    S2 = np.sum(np.multiply(maskY,img[i:i+n,j:j+n]))
                    newImage[i+1,j+1] = np.sqrt(np.power(S1,2)+np.power(S2,2))
        newImage *= 255.0 / newImage.max()            
        cv2.imwrite("Edge.png",newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png"))   
         
    def padding(self,img,n):
        self.R,self.C = img.shape
        imgAfterPadding = np.zeros((self.R+self.n-1,self.C+self.n-1))
        imgAfterPadding[1:1+self.R,1:1+self.C] = img.copy()
        return(imgAfterPadding)

    def paddingGeneral(self, desiredSize,img, n, backColor):
        # desired size img
        row, col = desiredSize.shape
        if backColor == 'w':
            imgAfterPadding = np.ones((row, col))
        elif backColor == 'b':
            imgAfterPadding = np.zeros((row, col))
        #center offset
        xx = (row-n) //2
        yy = (col-n) //2
        imgAfterPadding[xx:xx+n, yy:yy+n] = img.copy()
        return (imgAfterPadding)

    def getPicrures(self):
        path, extention = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            self.image = cv2.imread(path)
            self.grayImg = self.rgb2gray(self.image)
            self.padded = self.padding(self.grayImg,self.n)
            # self.paddingGeneral(self.grayImg,[[1,1,1],[0,0,0],[-1,-1,-1]] , 3,'w')

            self.ui.Input1.setPixmap(QPixmap(path))
    def getHistogram(self, img, type):

        # intenisties = np.arange(256)
        # histo = np.bincount(img[2], minlength=256)
        colors = ('r', 'g', 'b')
        for idx, color in enumerate(colors):
            histo, bins_edges = np.histogram(img[:, :, idx], bins=256, range=(0, 256))
            self.ui.graphicsView.setBackground('w')

            if type == 'c':
                histo = np.cumsum(histo)
            self.ui.graphicsView.plot(bins_edges[0:-1], histo, pen=color)

    def freqFilter(self, img, filterType):
        img_fshift = self.fourrier(img)
        # magnitude_spectrum = 20 * np.log(np.abs(img_fshift))

        # avg filter low pass fft
        if filterType == "a":
            mask = np.ones((3,3),np.float32)/9
            paddedMask = self.paddingGeneral(img,mask,3, 'b')
            maskFFT = self.fourrier(paddedMask)
            resultImg = maskFFT * img_fshift
            newImg = self.inverseFourrier(resultImg)
        # perwit high pass fft
        if filterType == "p":
            maskX = [[-1,0,1],[-1,0,1],[-1,0,1]]
            maskY = [[1,1,1],[0,0,0],[-1,-1,-1]]
            paddedMaskX = self.paddingGeneral(img, maskX, 3, 'w')
            paddedMaskY = self.paddingGeneral(img, maskY, 3, 'w')
            maskXFFT = self.fourrier(paddedMaskX)
            maskYFFT = self.fourrier(paddedMaskY)
            resultImgX = maskXFFT * img_fshift
            resultImgY = maskYFFT * img_fshift
            newImgX = self.inverseFourrier(resultImgX)
            newImgY = self.inverseFourrier(resultImgY)
            filteredImg = np.sqrt(np.power(newImgX, 2) + np.power(newImgY, 2))
            newImg *= 255.0 / filteredImg.max()

        cv2.imwrite("fourrierTest.png", newImg)
        self.ui.output1.setPixmap(QPixmap("./fourrierTest.png"))

        # self.ui.graphicsView.image(magnitude_spectrum)
        print("fft then send it to filter func")
    def fourrier(self,img):
        fourrier = np.fft.fft2(img)
        fshift = np.fft.fftshift(fourrier)
        return fshift
    def inverseFourrier(self, fourrImg):
        img_back = np.fft.ifftshift(fourrImg)
        img_back = np.fft.ifft2(img_back)
        img_back = np.abs(img_back)
        return img_back

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()        