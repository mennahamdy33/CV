# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ImageView.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1001, 654)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Myanmar Text")
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.filter = QtWidgets.QWidget()
        self.filter.setObjectName("filter")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.filter)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.loadTab1 = QtWidgets.QPushButton(self.filter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadTab1.sizePolicy().hasHeightForWidth())
        self.loadTab1.setSizePolicy(sizePolicy)
        self.loadTab1.setMaximumSize(QtCore.QSize(151, 111))
        self.loadTab1.setObjectName("loadTab1")
        self.horizontalLayout_2.addWidget(self.loadTab1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.inputTab1 = QtWidgets.QLabel(self.filter)
        self.inputTab1.setFrameShape(QtWidgets.QFrame.Box)
        self.inputTab1.setObjectName("inputTab1")
        self.horizontalLayout.addWidget(self.inputTab1)
        self.outputTab1 = QtWidgets.QLabel(self.filter)
        self.outputTab1.setFrameShape(QtWidgets.QFrame.Box)
        self.outputTab1.setObjectName("outputTab1")
        self.horizontalLayout.addWidget(self.outputTab1)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.groupBox = QtWidgets.QGroupBox(self.filter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 175))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.selectTab1 = QtWidgets.QComboBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.selectTab1.sizePolicy().hasHeightForWidth())
        self.selectTab1.setSizePolicy(sizePolicy)
        self.selectTab1.setMinimumSize(QtCore.QSize(500, 50))
        self.selectTab1.setMaximumSize(QtCore.QSize(500, 50))
        self.selectTab1.setObjectName("selectTab1")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.selectTab1.addItem("")
        self.horizontalLayout_11.addWidget(self.selectTab1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 80))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 80))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.minText = QtWidgets.QLabel(self.groupBox_2)
        self.minText.setObjectName("minText")
        self.horizontalLayout_8.addWidget(self.minText)
        self.min = QtWidgets.QLineEdit(self.groupBox_2)
        self.min.setObjectName("min")
        self.horizontalLayout_8.addWidget(self.min)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.maxText = QtWidgets.QLabel(self.groupBox_2)
        self.maxText.setObjectName("maxText")
        self.horizontalLayout_9.addWidget(self.maxText)
        self.max = QtWidgets.QLineEdit(self.groupBox_2)
        self.max.setObjectName("max")
        self.horizontalLayout_9.addWidget(self.max)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_9)
        self.gridLayout.addLayout(self.horizontalLayout_10, 0, 0, 1, 1)
        self.horizontalLayout_11.addWidget(self.groupBox_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_11, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.gridLayout_6.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.tabWidget.addTab(self.filter, "")
        self.histogram = QtWidgets.QWidget()
        self.histogram.setObjectName("histogram")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.histogram)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.loadTab2 = QtWidgets.QPushButton(self.histogram)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadTab2.sizePolicy().hasHeightForWidth())
        self.loadTab2.setSizePolicy(sizePolicy)
        self.loadTab2.setMinimumSize(QtCore.QSize(151, 111))
        self.loadTab2.setMaximumSize(QtCore.QSize(151, 111))
        self.loadTab2.setObjectName("loadTab2")
        self.verticalLayout_5.addWidget(self.loadTab2)
        self.gray = QtWidgets.QPushButton(self.histogram)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gray.sizePolicy().hasHeightForWidth())
        self.gray.setSizePolicy(sizePolicy)
        self.gray.setMinimumSize(QtCore.QSize(151, 111))
        self.gray.setMaximumSize(QtCore.QSize(151, 111))
        self.gray.setObjectName("gray")
        self.verticalLayout_5.addWidget(self.gray)
        self.color = QtWidgets.QPushButton(self.histogram)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.color.sizePolicy().hasHeightForWidth())
        self.color.setSizePolicy(sizePolicy)
        self.color.setMinimumSize(QtCore.QSize(151, 111))
        self.color.setMaximumSize(QtCore.QSize(151, 111))
        self.color.setObjectName("color")
        self.verticalLayout_5.addWidget(self.color)
        self.pushButton = QtWidgets.QPushButton(self.histogram)
        self.pushButton.setMinimumSize(QtCore.QSize(151, 111))
        self.pushButton.setMaximumSize(QtCore.QSize(151, 111))
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_5.addWidget(self.pushButton)
        self.horizontalLayout_5.addLayout(self.verticalLayout_5)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.inputTab2 = QtWidgets.QLabel(self.histogram)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inputTab2.sizePolicy().hasHeightForWidth())
        self.inputTab2.setSizePolicy(sizePolicy)
        self.inputTab2.setFrameShape(QtWidgets.QFrame.Box)
        self.inputTab2.setObjectName("inputTab2")
        self.horizontalLayout_4.addWidget(self.inputTab2)
        self.outputTab2 = QtWidgets.QLabel(self.histogram)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.outputTab2.sizePolicy().hasHeightForWidth())
        self.outputTab2.setSizePolicy(sizePolicy)
        self.outputTab2.setFrameShape(QtWidgets.QFrame.Box)
        self.outputTab2.setObjectName("outputTab2")
        self.horizontalLayout_4.addWidget(self.outputTab2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.inputHistogram = PlotWidget(self.histogram)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inputHistogram.sizePolicy().hasHeightForWidth())
        self.inputHistogram.setSizePolicy(sizePolicy)
        self.inputHistogram.setFrameShape(QtWidgets.QFrame.Box)
        self.inputHistogram.setObjectName("inputHistogram")
        self.horizontalLayout_3.addWidget(self.inputHistogram)
        self.outputHistogram = PlotWidget(self.histogram)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.outputHistogram.sizePolicy().hasHeightForWidth())
        self.outputHistogram.setSizePolicy(sizePolicy)
        self.outputHistogram.setFrameShape(QtWidgets.QFrame.Box)
        self.outputHistogram.setObjectName("outputHistogram")
        self.horizontalLayout_3.addWidget(self.outputHistogram)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.gridLayout_3.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
        self.tabWidget.addTab(self.histogram, "")
        self.hybrid = QtWidgets.QWidget()
        self.hybrid.setObjectName("hybrid")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.hybrid)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.load1Tab3 = QtWidgets.QPushButton(self.hybrid)
        self.load1Tab3.setMinimumSize(QtCore.QSize(151, 111))
        self.load1Tab3.setMaximumSize(QtCore.QSize(151, 111))
        self.load1Tab3.setObjectName("load1Tab3")
        self.verticalLayout_3.addWidget(self.load1Tab3)
        self.load2tab3 = QtWidgets.QPushButton(self.hybrid)
        self.load2tab3.setMinimumSize(QtCore.QSize(151, 111))
        self.load2tab3.setMaximumSize(QtCore.QSize(151, 111))
        self.load2tab3.setObjectName("load2tab3")
        self.verticalLayout_3.addWidget(self.load2tab3)
        self.hybrid1 = QtWidgets.QPushButton(self.hybrid)
        self.hybrid1.setMinimumSize(QtCore.QSize(151, 111))
        self.hybrid1.setMaximumSize(QtCore.QSize(151, 111))
        self.hybrid1.setObjectName("hybrid1")
        self.verticalLayout_3.addWidget(self.hybrid1)
        self.horizontalLayout_7.addLayout(self.verticalLayout_3)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.input1Tab3 = QtWidgets.QLabel(self.hybrid)
        self.input1Tab3.setFrameShape(QtWidgets.QFrame.Box)
        self.input1Tab3.setObjectName("input1Tab3")
        self.verticalLayout_4.addWidget(self.input1Tab3)
        self.input2Tab3 = QtWidgets.QLabel(self.hybrid)
        self.input2Tab3.setFrameShape(QtWidgets.QFrame.Box)
        self.input2Tab3.setObjectName("input2Tab3")
        self.verticalLayout_4.addWidget(self.input2Tab3)
        self.horizontalLayout_6.addLayout(self.verticalLayout_4)
        self.outputTab3 = QtWidgets.QLabel(self.hybrid)
        self.outputTab3.setFrameShape(QtWidgets.QFrame.Box)
        self.outputTab3.setObjectName("outputTab3")
        self.horizontalLayout_6.addWidget(self.outputTab3)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_6)
        self.gridLayout_4.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        self.tabWidget.addTab(self.hybrid, "")
        self.gridLayout_5.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1001, 22))
        self.menubar.setObjectName("menubar")
        self.menuExit = QtWidgets.QMenu(self.menubar)
        self.menuExit.setObjectName("menuExit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuExit.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadTab1.setText(_translate("MainWindow", "Load Image"))
        self.inputTab1.setText(_translate("MainWindow", "Input Image"))
        self.outputTab1.setText(_translate("MainWindow", "Output Image"))
        self.groupBox.setTitle(_translate("MainWindow", "Filters"))
        self.selectTab1.setItemText(0, _translate("MainWindow", "Select Filter"))
        self.selectTab1.setItemText(1, _translate("MainWindow", "Salt And Pepper"))
        self.selectTab1.setItemText(2, _translate("MainWindow", "Average Filter"))
        self.selectTab1.setItemText(3, _translate("MainWindow", "Gaussian Filter"))
        self.selectTab1.setItemText(4, _translate("MainWindow", "Median Filter"))
        self.selectTab1.setItemText(5, _translate("MainWindow", "Sobel Filter"))
        self.selectTab1.setItemText(6, _translate("MainWindow", "Roberts Filter"))
        self.selectTab1.setItemText(7, _translate("MainWindow", "Prewitt Filter"))
        self.selectTab1.setItemText(8, _translate("MainWindow", "Normalization"))
        self.selectTab1.setItemText(9, _translate("MainWindow", "Equalization"))
        self.selectTab1.setItemText(10, _translate("MainWindow", "Local Thresholding"))
        self.selectTab1.setItemText(11, _translate("MainWindow", "Global Thresholding"))
        self.selectTab1.setItemText(12, _translate("MainWindow", "Low Frequency Filter"))
        self.selectTab1.setItemText(13, _translate("MainWindow", "High Frequency Filter"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Add Parameters"))
        self.minText.setText(_translate("MainWindow", "Min:"))
        self.maxText.setText(_translate("MainWindow", "Max:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.filter), _translate("MainWindow", "Filters"))
        self.loadTab2.setText(_translate("MainWindow", "Load"))
        self.gray.setText(_translate("MainWindow", "Gray"))
        self.color.setText(_translate("MainWindow", "Color"))
        self.pushButton.setText(_translate("MainWindow", "Cumulative Color"))
        self.inputTab2.setText(_translate("MainWindow", "Input Image"))
        self.outputTab2.setText(_translate("MainWindow", "Output Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.histogram), _translate("MainWindow", "Histograms"))
        self.load1Tab3.setText(_translate("MainWindow", "Loag Image 1"))
        self.load2tab3.setText(_translate("MainWindow", "Load Image 2"))
        self.hybrid1.setText(_translate("MainWindow", "Make Hybride"))
        self.input1Tab3.setText(_translate("MainWindow", "Input Image 1"))
        self.input2Tab3.setText(_translate("MainWindow", "Input Image 2"))
        self.outputTab3.setText(_translate("MainWindow", "Output Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.hybrid), _translate("MainWindow", "Hybrid"))
        self.menuExit.setTitle(_translate("MainWindow", "Exit"))

from pyqtgraph import PlotWidget
