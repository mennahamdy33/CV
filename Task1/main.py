from PyQt5 import QtWidgets
from ImageViewer import Ui_MainWindow
import sys
import numpy as np
import pandas as pd

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionExit.triggered.connect(exit)
        self.ui.actionInput.triggered.connect(self.getPicrures)
        
    def getPicrures (self):
        pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()        