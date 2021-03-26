from PyQt5 import QtWidgets,QtGui
from ImageViewer import Ui_MainWindow
import sys
import numpy as np
import pandas as pd
import cv2
QPixmap = QtGui.QPixmap

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.padded =[]
        self.R=0;
        self.C=0;
        self.n=3
        self.ui.actionExit.triggered.connect(exit)
        self.ui.actionInput.triggered.connect(self.getPicrures)
        self.ui.filter.clicked.connect(lambda:self.AvgFilter(self.padded,self.R,self.C,self.n))
        self.ui.edge.activated.connect(self.chooseEdge)

    def chooseEdge(self):
        if str(self.ui.edge.currentText())=="Edge In X":
            maskX = [[-1,0,1],[-2,0,2],[-1,0,1]]
            self.SobelFilter(self.padded,self.R,self.C,self.n,maskX)
        if str(self.ui.edge.currentText())=="Edge In Y":
            maskY = [[1,2,1],[0,0,0],[-1,-2,-1]]    
            self.SobelFilter(self.padded,self.R,self.C,self.n,maskY)

    def AvgFilter(self,img,R,C,n):
        mask = np.ones((3,3),np.float32)/9
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    newImage[i-1,j-1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
        cv2.imwrite("Filtered.png",newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Filtered.png"))

    def SobelFilter(self,img,R,C,n,mask):
        newImage = np.zeros((R+n-1,C+n-1))
        for i in range(1,R-2):
            for j in range(1,C-2):
                    newImage[i-1,j-1] = np.sum(np.multiply(mask,img[i:i+n,j:j+n]))
        cv2.imwrite("Edge.png",newImage)
        self.ui.output1.setPixmap(QPixmap("G:/SBME/CV/Tasks/CV/Task1/Edge.png")) 

    def padding(self,img,n):
        self.R,self.C= img.shape
        imgAfterPadding = np.zeros((self.R+self.n-1,self.C+self.n-1))
        imgAfterPadding[1:1+self.R,1:1+self.C] = img.copy()
        return(imgAfterPadding)       
        
    def getPicrures (self):
        path,extention = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "") 
        if path == "":
            pass
        else:
            image = cv2.imread(path,0)
            self.padded = self.padding(image,self.n)
            self.ui.Input1.setPixmap(QPixmap(path))

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()        