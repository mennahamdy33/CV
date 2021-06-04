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
from tkinter import Toplevel, Label
import matplotlib.pyplot as plt #plot import
import matplotlib.colors  #color import
import numpy as np  #importing numpy
from PIL import Image #importing PIL to read all kind of images
from PIL import ImageTk
import glob
import cv2
import os
import math
from icecream import ic
import sklearn.metrics as metrics
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.value =0 
        self.r = 0
        self.c = 0
        self.V = 0
        self.ui.loadTab1.clicked.connect(lambda: self.getPicrures(1))
        self.ui.faceDetection.clicked.connect(self.FaceDetection)
        self.ui.loadTraing.clicked.connect(self.reading_faces_and_displaying)
        self.ui.makeEigenfaces.clicked.connect(self.Eigen)
        self.ui.loadTest.clicked.connect(self.reading_test_images)
        self.ui.slider.valueChanged.connect(self.reading_test_images)
        self.ui.match.clicked.connect(self.function)
        self.ui.checkError.clicked.connect(self.ROC)
        self.test_list = []
        self.substract_mean_from_original = []
        self.train_list = []
        # shofy enty 3ayza el value deh fen
        self.threshold= ''


    def slider(self):
        self.value = self.ui.slider.value()

    def error_for_k(self,k,test_from_mean,V,substract_mean_from_original,train_list,test_list):
        TPcount = 0
        TNcount = 0
        PP=[]
        TP = []
        eigen_weights = np.dot(V[:k, :],substract_mean_from_original.T)
        for i in range(test_from_mean.shape[0]):

            P=''
            N=''
            test_weight = np.dot(V[:k, :],test_from_mean[i:i + 1,:].T)
            distances_euclidian = np.sum((eigen_weights - test_weight) ** 2, axis=0)
            image_closest = np.argmin(np.sqrt(distances_euclidian))
            x = test_list[i]

            if (distances_euclidian[image_closest] <= int(self.threshold , base =10)):
                P = train_list[image_closest]
                PP.append(1)

            else:
                PP.append(0)
                N = train_list[image_closest]


            if (x == P) :
                TP.append(1)
                TPcount += 1
            else:
                TP.append(0)


            if (x != N) and (N != '') :
                TNcount += 1

        accuracy = (TPcount + TNcount) / (test_from_mean.shape[0]+1)

        return TP,PP,accuracy

    def ROC(self):
        k = self.k
        self.ui.rocCurve.clear()
        TP,PP,accuracy = self.error_for_k(k,self.test_from_mean,
                        self.V,self.substract_mean_from_original,
                        self.train_list,self.test_list)
        self.ui.errorText.setText(str(accuracy))
        self.ui.rocCurve.setBackground('w')

        styles = { 'font-size': '20px'}
        self.ui.rocCurve.setLabel('left', 'True Positive Rate', **styles)
        self.ui.rocCurve.setLabel('bottom', 'False Positive Rate', **styles)

        self.ui.rocCurve.setXRange(0, 1, padding=0.5)
        self.ui.rocCurve.setYRange(0, 1, padding=0.1)
        pen1 = pg.mkPen(color='b',width=2, style=QtCore.Qt.DashLine)
        pen2 = pg.mkPen(color='r', width=5)

        self.ui.rocCurve.addLegend()
        for x in range(len(TP)):

            fpr, tpr, thresholds = metrics.roc_curve(TP, PP)

            self.ui.rocCurve.plot(fpr,tpr,pen = pen2, name="ROC Curve for class"+str(x+1))
        self.ui.rocCurve.plot([0,1],[0,1],pen = pen1 )

    def read_pgm(self,pgmf):
        header = pgmf.readline()
        assert header[:2] == b'P5'
        (width, height) = [int(i) for i in header.split()[1:3]]
        depth = int(header.split()[3])
        assert depth <= 65535

        raster = []
        for y in range(height):
            row = []
            for y in range(width):
                low_bits = ord(pgmf.read(1))
                row.append(low_bits+255*ord(pgmf.read(1)))
            raster.append(row)
        return raster

    def reading_faces_and_displaying(self):
        face_array = []
        for face_images in glob.glob('./dataSet/Train/*.jpg'): # assuming jpg
            a1 = face_images
            _,a1 = a1.split('\\')

            a1,_=a1.split('_', maxsplit=1)

            self.train_list.append(a1)
            face_image=Image.open(face_images)
            face_image = np.asarray(face_image,dtype=float)/255.0
            
            face_array.append(face_image)
        face_array=np.asarray(face_array)
        n ,self.r,self.c=face_array.shape
        self.ui.noImages.setText(str(len(face_array)))

        self.ui.dbName.setText(str("AT&T"))

        return face_array

    def performing_pca(self,face_array):
        mean = np.mean(face_array, 0)
        flatten_Array = []
        for x in range(len(face_array)):
            flat_Array = face_array[x].flatten()
            flatten_Array.append(flat_Array)
        flatten_Array = np.asarray(flatten_Array)
        mean = mean.flatten()
    
        return mean,flatten_Array,  

    def Eigen(self):
        face_array=self.reading_faces_and_displaying()
        mean,flatten_Array=self.performing_pca(face_array) # eigen_values,eigen_vectors
        substract_mean_from_original = np.subtract(flatten_Array, mean)
        U, s, self.V = np.linalg.svd(substract_mean_from_original, full_matrices=False)
        self.k = 15
        self.ui.selectedEigenfaces.setText(str(self.k))
        return (self.k,face_array,mean,substract_mean_from_original,self.V)

    def class_face(self,k,test_from_mean,test_flat_images,V,substract_mean_from_original,face_array):

        eigen_weights = np.dot(V[:k, :],substract_mean_from_original.T)
        # for i in range(test_from_mean.shape[0]):
        test_weight = np.dot(V[:k, :],test_from_mean[self.value:self.value + 1,:].T)
        distances_euclidian = np.sum((eigen_weights - test_weight) ** 2, axis=0)
        image_closest = np.argmin(np.sqrt(distances_euclidian))
        # to_plot=np.reshape(test_flat_images[self.value,:], (425,425))
        # cv2.imshow('to_plot', to_plot)
        # cv2.waitKey(0)
        self.ui.parameters.setText(str(round(distances_euclidian[image_closest],3)))
        if (distances_euclidian[image_closest] <= int(self.threshold, base = 10)):
            self.ui.parameters.setText(str(round(distances_euclidian[image_closest],3)))
            # cv2.imshow("face_array[image_closest,:,:]", face_array[image_closest,:,:])
            # cv2.waitKey(0)
            cv2.imwrite("./Images/test1.jpg",(face_array[image_closest,:,:]*255))
            w = self.ui.bestMatch.width()
            h = self.ui.bestMatch.height()
            self.ui.bestMatch.setPixmap(QPixmap("./Images/test1.jpg").scaled(w, h, QtCore.Qt.KeepAspectRatio))
        else :
            self.ui.bestMatch.setText("NO MATCH")
   
    def returning_vector(self,test_images):
        flat_test_Array = []
        for x in range(len(test_images)):
            flat_Array = test_images[x].flatten()
            flat_test_Array.append(flat_Array)
        flat_test_Array = np.asarray(flat_test_Array)
        return flat_test_Array

    def reading_test_images(self):
        self.value = self.ui.slider.value()
        
        test_images=[]
        for images in glob.glob('./dataSet/Test/*.jpg'):  # assuming jpg
            _,a1 = images.split('\\')
            a1,_= a1.split('_', maxsplit=1)  
            self.test_list.append(a1)      
            test_ = Image.open(images)
            test_facess = np.asarray(test_, dtype=float)
            test_faces = test_facess /255
            test=(self.r,self.c,3)
            test2 = (self.r,self.c)
            if test_faces.shape == test:
                test_faces=test_faces[:,:,0]
                test_images.append(test_faces)
            elif test_faces.shape == test2  :
                 test_images.append(test_faces)
            else:
                pass
                # test_images.append(test_faces)

        self.ui.slider.setMinimum(0)
        self.ui.slider.setMaximum(len(test_images)-1)
        self.ui.tesImageText.setText("Test Image {}/{}".format(self.value+1,len(test_images)) )
        cv2.imwrite("./Images/test.jpg",(test_images[self.value]*255))
        w = self.ui.testImage.width()
        h = self.ui.testImage.height()
        self.ui.testImage.setPixmap(QPixmap("./Images/test.jpg").scaled(w, h, QtCore.Qt.KeepAspectRatio))
        
        flat_test_Array=self.returning_vector(test_images)
        test_images=np.asarray(test_images)
        return flat_test_Array,test_images

    def function(self):
        self.threshold= self.ui.lineEdit.text()
        k,face_array,mean,self.substract_mean_from_original,V = self.Eigen()
        test_flat_images,test_images=self.reading_test_images()
        self.test_from_mean = np.subtract(test_flat_images,mean)

        self.class_face(k,self.test_from_mean,test_flat_images,V,self.substract_mean_from_original,face_array)

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
                
        cv2.imwrite("./Images/Detection_Output.jpg",image_copy)        
        w = self.ui.outputTab1.width()
        h = self.ui.outputTab1.height()
        self.ui.outputTab1.setPixmap(QPixmap("./Images/Detection_Output.jpg").scaled(w, h, QtCore.Qt.KeepAspectRatio))        

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
