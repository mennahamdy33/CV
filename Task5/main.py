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

import math

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.value =0 
        self.ui.loadTab1.clicked.connect(lambda: self.getPicrures(1))
        self.ui.faceDetection.clicked.connect(self.FaceDetection)
        self.ui.loadTraing.clicked.connect(self.reading_faces_and_displaying)
        self.ui.makeEigenfaces.clicked.connect(self.Eigen)
        self.ui.loadTest.clicked.connect(self.reading_test_images)
        self.ui.slider.valueChanged.connect(self.reading_test_images)
        self.ui.match.clicked.connect(self.function)

    def slider(self):
        self.value = self.ui.slider.value()

    def reading_faces_and_displaying(self):
        face_array = []
        for face_images in glob.glob('./Eigenfaces/Train/*.jpg'): # assuming jpg
            face_image=Image.open(face_images)
            face_image = np.asarray(face_image,dtype=float)/255.0
            face_array.append(face_image)
        face_array=np.asarray(face_array)
        self.ui.noImages.setText(str(len(face_array)))
        self.ui.noPersons.setText(str(len(face_array)))
        return face_array

    def performing_pca(self,face_array):
        mean = np.mean(face_array, 0)
        flatten_Array = []
        for x in range(len(face_array)):
            flat_Array = face_array[x].flatten()
            flatten_Array.append(flat_Array)
        flatten_Array = np.asarray(flatten_Array)
        mean = mean.flatten()
        # flatten_Array=flatten_Array.T
        #print(flatten_Array.shape)
        #face_array = face_array.flatten()
        # mean=mean.T
        #substract_mean_from_original = np.subtract(flatten_Array, mean)
        # transpose_substract_mean_from_original=substract_mean_from_original.T
        # eigen_faces=displaying_eigen_faces(face_array,mean)
        #covariance_matrix = np.cov(substract_mean_from_original)
        #eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        return mean,flatten_Array,  

    def Eigen(self):
        face_array=self.reading_faces_and_displaying()
        mean,flatten_Array=self.performing_pca(face_array) # eigen_values,eigen_vectors
        substract_mean_from_original = np.subtract(flatten_Array, mean)
        U, s, V = np.linalg.svd(substract_mean_from_original, full_matrices=False)
        self.ui.selectedEigenfaces.setText(str(len(V)))
        print("marwa khody balek",len(V))
        Eigen_faces=[]
        for x in range(V.shape[0]):
            fig=np.reshape(V[x],(425,425))
            Eigen_faces.append(fig)
        return Eigen_faces,face_array,mean,substract_mean_from_original,V

    def class_face(self,k,test_from_mean,test_flat_images,V,substract_mean_from_original,face_array):
        eigen_weights = np.dot(V[:k, :],substract_mean_from_original.T)
        threshold = 6000

        # for i in range(test_from_mean.shape[0]):
        test_weight = np.dot(V[:k, :],test_from_mean[self.value:self.value + 1,:].T)
        distances_euclidian = np.sum((eigen_weights - test_weight) ** 2, axis=0)
        image_closest = np.argmin(np.sqrt(distances_euclidian))
        # to_plot=np.reshape(test_flat_images[self.value,:], (425,425))
        # cv2.imshow('to_plot', to_plot)
        # cv2.waitKey(0)
        if (distances_euclidian[image_closest] <= threshold):
            # cv2.imshow("face_array[image_closest,:,:]", face_array[image_closest,:,:])
            # cv2.waitKey(0)
            cv2.imwrite("./Images/test1.jpg",(face_array[image_closest,:,:]*255))
            self.ui.bestMatch.setPixmap(QPixmap("./Images/test1.jpg"))
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
        for images in glob.glob('./Eigenfaces/Test/*.jpg'):  # assuming jpg
            test_ = Image.open(images)
            test_facess = np.asarray(test_, dtype=float)
            test_faces = test_facess /255
            #test_faces = test_faces.convert('L')  #int
            #test_faces = np.asarray(test_faces, dtype=float) / 255.0  # Normalize the image to be between 0 to 1
            test=(425,425,3)
            if test_faces.shape == test:
                test_faces=test_faces[:,:,0]
                test_images.append(test_faces)
            else:
                test_images.append(test_faces)

        self.ui.slider.setMinimum(0)
        self.ui.slider.setMaximum(len(test_images)-1)
        print(self.value)
        self.ui.tesImageText.setText("Test Image {}/{}".format(self.value+1,len(test_images)) )
        cv2.imwrite("./Images/test.jpg",(test_images[self.value]*255))
        self.ui.testImage.setPixmap(QPixmap("./Images/test.jpg"))
        #print(test_images[25].shape)
        #print(test_images[25].shape[0])
        #if test_images[25].shape[0]:
        #    print("a")
        flat_test_Array=self.returning_vector(test_images)
        test_images=np.asarray(test_images)
        return flat_test_Array,test_images

    def function(self):
        Eigen_faces,face_array,mean,substract_mean_from_original,V = self.Eigen()
        test_flat_images,test_images=self.reading_test_images()
        test_from_mean=np.subtract(test_flat_images,mean)

        k=15
        print("FACES FOR K=2")
        self.class_face(k,test_from_mean,test_flat_images,V,substract_mean_from_original,face_array)

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
