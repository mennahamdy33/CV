from PyQt5 import QtWidgets,QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
from imageview import Ui_MainWindow

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
