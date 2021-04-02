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
            self.Normalization()
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
        # feh akher el function ektby keda "self.ui.groupBox_2.hide()"
        pass
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
        pass
    def GlobalThresholding(self):
        pass
    def LowFreqFilter(self):
        self.freqFilter(self.grayImg,"a")
    def HighFreqFilter(self):
        pass
    def gaussianFilter(self,img):
        mean = np.mean(img)
        std = np.std(img)
        FilteredImage = (img-mean)/std
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
            self.padded = self.padding(self.grayImg,self.n)
            if (tab == 1):
                self.ui.inputTab1.setPixmap(QPixmap(path))
            elif (tab == 2):
                self.ui.inputTab2.setPixmap(QPixmap(path))
                # test image

                r, g, b = cv2.split(self.image)
                # spliting b,g,r and getting differences between them
                r_g = np.count_nonzero(abs(r - g))
                r_b = np.count_nonzero(abs(r - b))
                g_b = np.count_nonzero(abs(g - b))
                diff_sum = float(r_g + r_b + g_b)
                # finding ratio of diff_sum with respect to size of image
                ratio = diff_sum / self.image.size
                if ratio > 0.005:
                    label = 'color'

                else:
                    label = 'grey'
                    self.getHistogram(self.image,label)
                print(label)
            elif (tab == 3):
                self.ui.input1Tab3.setPixmap(QPixmap(path))
                self.LowCompImage = self.padded
            elif (tab == 4):
                self.ui.input2Tab3.setPixmap(QPixmap(path))
                self.HighCompImage = self.padded

            # self.paddingGeneral(self.grayImg,[[1,1,1],[0,0,0],[-1,-1,-1]] , 3,'w')

    def getHistogram(self, img, type):
        if type == 'color':
            # intenisties = np.arange(256)
            # histo = np.bincount(img[2], minlength=256)
            colors = ('r', 'g', 'b')
            for idx, color in enumerate(colors):
                histo, bins_edges = np.histogram(img[:, :, idx], bins=256, range=(0, 256))
                self.ui.inputHistogram.setBackground('w')
                if type == 'c':
                    histo = np.cumsum(histo)
                self.ui.inputHistogram.plot(bins_edges[0:-1], histo, pen=color)
        elif type == 'grey':
            histo, bins_edges = np.histogram(img[:, :], bins=256, range=(0, 256))
            print(histo)
            self.ui.inputHistogram.plot(bins_edges[0:-1], histo)
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
        self.ui.outputTab1.setPixmap(QPixmap("./fourrierTest.png"))

        # self.ui.graphicsView.image(magnitude_spectrum)
        print("fft then send it to filter func")

    def fourrier(self, img):
        fourrier = np.fft.fft2(img)
        fshift = np.fft.fftshift(fourrier)
        return fshift

    def inverseFourrier(self, fourrImg):
        f_ishift = np.fft.ifftshift(fourrImg)
        img_back = cv2.idft(f_ishift)
        # img_back = np.fft.ifftshift(fourrImg)
        # img_back = np.fft.ifft2(img_back)
        img_back = np.abs(img_back)
        return img_back




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
