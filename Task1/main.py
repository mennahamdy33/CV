from PyQt5 import QtWidgets,QtGui
from ImageView import Ui_MainWindow
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
        self.n=3
        self.precent=0.1
        self.alpha = 0.8
        self.minIntensity = 0
        self.maxIntensity = 255
        self.Th = 150 
        self.ui.groupBox_2.hide()
        self.ui.selectTab1.activated.connect(self.chooseFilter)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.loadTab1.clicked.connect(lambda:self.getPicrures(1))
        self.ui.loadTab2.clicked.connect(lambda:self.getPicrures(2))
        self.ui.load1Tab3.clicked.connect(lambda:self.getPicrures(3))
        self.ui.load2tab3.clicked.connect(lambda:self.getPicrures(4))
        self.ui.hybrid1.clicked.connect(self.Hybrid)

        # self.ui.histogram.clicked.connect(lambda: self.getHistogram(self.image, 'c'))
        # self.ui.freqDominFilter.clicked.connect(lambda: self.freqFilter(self.grayImg))

    def Hybrid(self):
        imageSmoothed = self.AvgFilter(self.LowCompImage,3)
        self.ui.outputTab1.setText("Output Image")
        # imageSmoothed = self.gaussianFilter(self.image)
        imageSharped = self.LaplacianFilter(self.HighCompImage,3)
        outputImage = (imageSmoothed * (1-self.alpha)) + (imageSharped*self.alpha)
        cv2.imwrite(r"./images/HybridImage.png",outputImage)
        self.ui.outputTab3.setPixmap(QPixmap(r"./images/HybridImage.png"))

    def rgb2gray(self, rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    def salt_pepper_noise(self,img, percent):
        img_noisy = np.zeros(img.shape)
        salt_pepper = np.random.random(img.shape)  # Uniform distribution
        cleanPixels_ind = salt_pepper > percent
        NoisePixels_ind = salt_pepper <= percent
        pepper = (salt_pepper <= (0.5 * percent))  # pepper < half percent
        salt = ((salt_pepper <= percent) & (salt_pepper > 0.5 * percent))
        img_noisy[cleanPixels_ind] = img[cleanPixels_ind]
        img_noisy[pepper] = 0
        img_noisy[salt] = 1
        cv2.imwrite(r"./images/noise.png", img_noisy)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/noise.png"))
        return (img_noisy)

    def chooseFilter(self):
        if str(self.ui.selectTab1.currentText())=="Salt And Pepper":
            self.noiseImage = self.salt_pepper_noise(self.padded,self.precent)
        if str(self.ui.selectTab1.currentText())=="Average Filter":
            self.FilterImage = self.AvgFilter(self.noiseImage,3)
        if str(self.ui.selectTab1.currentText())=="Gaussian Filter" :   
            self.FilterImage = self.gaussianFilter(self.noiseImage)
        if str(self.ui.selectTab1.currentText())=="Median Filter" :   
            self.FilterImage = self.medianFilter(self.noiseImage,3)
        if str(self.ui.selectTab1.currentText())=="Sobel Filter" :   
            maskX = [[-1,0,1],[-2,0,2],[-1,0,1]]
            maskY = [[1,2,1],[0,0,0],[-1,-2,-1]] 
            file = r"./images/SobelFilter.png"
            self.highFilter(self.FilterImage,3,maskX,maskY,file)
        if str(self.ui.selectTab1.currentText())=="Roberts Filter" : 
            maskX = [[1,0],[0,-1]]
            maskY = [[0,1],[-1,0]] 
            file = r"./images/RobertsFilter.png"
            self.highFilter(self.FilterImage,2,maskX,maskY,file)
        if str(self.ui.selectTab1.currentText())=="Prewitt Filter" :   
            maskX = [[-1,0,1],[-1,0,1],[-1,0,1]]
            maskY = [[1,1,1],[0,0,0],[-1,-1,-1]]
            file = r"./images/PrewittFilter.png"
            self.highFilter(self.FilterImage,3,maskX,maskY,file) 
        if str(self.ui.selectTab1.currentText())=="Normalization":
            self.ui.groupBox_2.show()
            # self.Normalization()
        if str(self.ui.selectTab1.currentText())=="Equalization":
            self.Equalization()
        if str(self.ui.selectTab1.currentText())=="Local Thresholding":
            self.LocalThresholding()
        if str(self.ui.selectTab1.currentText())=="Global Thresholding":
            self.GlobalThresholding()
        if str(self.ui.selectTab1.currentText())=="Low Frequency Filter" :   
            self.LowFreqFilter()
        if str(self.ui.selectTab1.currentText())=="High Frequency Filter" :   
            self.HighFreqFilter()

    def Normalization(self):
        img = self.grayImg
        minIntensity = np.int(self.ui.minText.text())
        maxIntensity= np.int(self.ui.maxText.text())
        img = ((img-minIntensity)*maxIntensity)/(maxIntensity-minIntensity)
        cv2.imwrite(r"./images/normalized.png", eq_img_array)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/normalized.png"))
    
    
    

    def Equalization(self):
        array = np.around(self.grayImg).astype(int)
        histo, bins_edges = np.histogram(array.flatten(), bins=256, range=(0, 256))
        chistogram_array = np.cumsum(histo)
        chistogram_array = chistogram_array * (255/chistogram_array.max())
        img_list = list(array.flatten())
        eq_img_list = [chistogram_array[p] for p in img_list]
        eq_img_array = np.reshape(np.asarray(eq_img_list), array.shape)
        cv2.imwrite(r"./images/equalized.png", eq_img_array)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/equalized.png"))

    def LocalThresholding(self):
        n = 5 # mask 5 x 5
        newImg = np.zeros((self.R,self.C))
        img = self.padding(self.grayImg,n)
        R,C = img.shape
        for i in range(R-n//2):
            for j in range(C-n//2):
                mask = np.average(img[i:i+5,i:i+5])
                newImg[i,j] = self.maxIntensity if mask >= self.Th else self.minIntensity
        return(newImg)
    

    def GlobalThresholding(self):
        img = self.grayImg
        newImg = np.zeros((self.R,self.C))
        for i in range(self.R):
            for j in range(self.C):
                newImg[i,j] = self.maxIntensity if img[i,j] >= self.Th else self.minIntensity
        return(newImg)

    def LowFreqFilter(self):
        self.freqFilter(self.grayImg,"a")
    def HighFreqFilter(self):
        self.freqFilter(self.grayImg, "p")
    def gaussianFilter(self,img):
        sigma =1
        FilteredImage = np.zeros(img.shape)
        for i in range(self.R):
            for j in range(self.C):
                FilteredImage[i,j] =1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(img[i,j])**2/(2*sigma**2))
        print(FilteredImage)
        cv2.imwrite(r"./images/GaussianFilter.png",FilteredImage)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/GaussianFilter.png"))
        return (FilteredImage)        

    def medianFilter(self,img,n):
        R,C = img.shape
        FilteredImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                mask = img[i:i+n,j:j+n]
                FilteredImage[i-1,j-1] = np.median(mask[:])
        cv2.imwrite(r"./images/MedianFilter.png",FilteredImage)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/MedianFilter.png"))
        return (FilteredImage)

    def LaplacianFilter(self,img,n):
        R,C = img.shape
        mask = [[0,-1,0],[-1,4,-1],[0,-1,0]]
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                newImage[i+1,j+1] = np.sum(np.sum(np.multiply(mask,img[i:i+n,j:j+n])))
        newImage *= 255.0 / newImage.max()
        cv2.imwrite(r"./images/LaplacianFilter.png",newImage)
        return newImage

    def AvgFilter(self,img,n):
        R,C = img.shape
        mask = np.ones((3,3),np.float32)/9
        FilteredImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    FilteredImage[i+1,j+1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
        cv2.imwrite(r"./images/AvgFilter.png",FilteredImage)
        self.ui.outputTab1.setPixmap(QPixmap(r"./images/AvgFilter.png"))
        return (FilteredImage)

    def highFilter(self,img,n,maskX,maskY,file):
        R,C = img.shape
        newImage = np.zeros((R+n-1,C+n-1))
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    S1 = np.sum(np.multiply(maskX,img[i:i+n,j:j+n]))
                    S2 = np.sum(np.multiply(maskY,img[i:i+n,j:j+n]))
                    newImage[i+1,j+1] = np.sqrt(np.power(S1,2)+np.power(S2,2))
        newImage *= 255.0 / newImage.max()
        cv2.imwrite(file,newImage)
        self.ui.outputTab1.setPixmap(QPixmap(file))

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

    def getPicrures(self, tab):
        path, extention = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            self.image = cv2.imread(path)
            self.grayImg = self.rgb2gray(self.image)
            cv2.imwrite("graylinda.png", self.grayImg)
            self.padded = self.padding(self.grayImg,self.n)
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

            # self.paddingGeneral(self.grayImg,[[1,1,1],[0,0,0],[-1,-1,-1]] , 3,'w')

    def getHistogram(self, img, type):

        # intenisties = np.arange(256)
        # histo = np.bincount(img[2], minlength=256)
        colors = ('r', 'g', 'b')
        for idx, color in enumerate(colors):
            histo, bins_edges = np.histogram(img[:, :, idx], bins=256, range=(0, 256))
            self.ui.inputHistogram.setBackground('w')
            if type == 'c':
                histo = np.cumsum(histo)
            self.ui.inputHistogram.plot(bins_edges[0:-1], histo, pen=color)

    def freqFilter(self, img, filterType):
        img_fshift = self.fourrier(img)
        # avg filter low pass fft
        if filterType == "a":
            mask = np.ones((3,3),np.float32)/9
            paddedMask = self.paddingGeneral(img,mask,3, 'b')
            maskFFT = self.fourrier(paddedMask)
            resultImg = maskFFT * img_fshift
            newImg = self.inverseFourrier(resultImg)
        # laplacian high pass fft
        if filterType == "p":
            mask = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
            paddedMask = self.paddingGeneral(img, mask, 3, 'b')
            maskFFT = self.fourrier(paddedMask)
            test = np.log(np.abs(maskFFT))
            cv2.imwrite("maskfft.png", test)
            resultImg = maskFFT * img_fshift
            newImg = self.inverseFourrier(resultImg)

        cv2.imwrite("fourrierTest.png", newImg)
        self.ui.outputTab1.setPixmap(QPixmap("./fourrierTest.png"))

        # self.ui.graphicsView.image(magnitude_spectrum)
        print("fft then send it to filter func")

    def fourrier(self, img):
        fourrier = np.fft.fft2(img)
        fshift = np.fft.fftshift(fourrier)
        return fourrier

    def inverseFourrier(self, fourrImg):
        # img_back = np.fft.ifftshift(fourrImg)
        img_back = np.fft.ifft2(fourrImg)
        img_back = np.fft.ifftshift(img_back)
        img_back = np.abs(img_back)
        return img_back




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
