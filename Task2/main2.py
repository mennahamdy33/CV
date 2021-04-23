from PyQt5 import QtWidgets, QtGui
from imageview import Ui_MainWindow
import sys
import numpy as np
import cv2
from scipy.ndimage.filters import convolve
QPixmap = QtGui.QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import matplotlib.pyplot as plt
from collections import defaultdict


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.menuExit.triggered.connect(exit)
        self.ui.loadEdgeImg.clicked.connect(lambda: self.getPicrures(5))
        self.ui.CannyEdgeBtn.clicked.connect(lambda :self.canny(0))
        self.ui.houghLineBtn.clicked.connect(self.line)
        self.ui.houghCircleBtn.clicked.connect(self.circle)

    def getPicrures(self, tab):
        path, extention = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "")
        if path == "":
            pass
        else:
            self.image = cv2.imread(path)
            self.grayImg = self.rgb2gray(self.image)
            cv2.imwrite(r"./images/grayPicture.png", self.grayImg)
            if (tab == 5):
                self.ui.InputTab4.setPixmap(QPixmap(path))

    def canny(self,flag):
        smoothedImg = self.gaussianFilter()
        mg, angle = self.highFilter(smoothedImg)
        nonMaxImg = self.non_max_suppression(mg, angle)
        thresh, weak, strong = self.threshold(nonMaxImg, 0.05, 0.09)
        img_final = self.hysteresis(thresh, weak, strong)
        cv2.imwrite(r".\images\cannyEdge.png", img_final)
        if flag == 0:
            self.ui.edgeView.setPixmap(QPixmap(r".\images\cannyEdge.png"))
        if flag == 1:
            return img_final
    def line(self):
        image_final = self.canny(1)
        hough_imgs = self.hough_line(image_final)
        self.show_lines(self.image, hough_imgs, 0.9995)
        self.ui.edgeView.setPixmap(QPixmap(r".\images\houghLine.png"))
    def show_lines(self,img, hough_img, percentile):
        fig, ax = plt.subplots()

        acc, thetas, rhos = hough_img

        lines = self.extract_lines(*hough_img, percentile)

        ax.imshow(img)
        ax.autoscale(False)
        plt.axis('off')
        for (rho, theta), val in lines.items():
            a = np.cos(theta)
            b = np.sin(theta)
            width = 1000
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + width * (-b))
            y1 = int(y0 + width * (a))
            x2 = int(x0 - width * (-b))
            y2 = int(y0 - width * (a))
            ax.plot((x1, x2), (y1, y2), '-r')

        plt.savefig(r'.\images\houghLine.png', bbox_inches='tight', pad_inches=0)

    def extract_lines(self,accumulator, thetas, rhos, percentile):
        lines = defaultdict()
        threshold = np.quantile(accumulator, percentile)
        for rho_idx in range(accumulator.shape[0]):
            for theta_idx in range(accumulator.shape[1]):
                if accumulator[rho_idx, theta_idx] > threshold:
                    theta = thetas[theta_idx]
                    rho = rhos[rho_idx]
                    lines[(rho, theta)] = accumulator[rho_idx, theta_idx]

        return lines

    def hough_line(self,image):

        Ny = image.shape[0]
        Nx = image.shape[1]
        Maxdist = int(np.round(np.sqrt(Nx ** 2 + Ny ** 2)))

        thetas = np.deg2rad(np.arange(-90, 90))

        rs = np.linspace(-Maxdist, Maxdist, 2 * Maxdist)

        accumulator = np.zeros((2 * Maxdist, len(thetas)))

        for y in range(Ny):
            for x in range(Nx):
                if image[y, x] > 0:
                    for k in range(len(thetas)):
                        r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                        accumulator[int(r) + Maxdist, k] += 1
        return accumulator, thetas, rs
    def circle(self):
        res = self.detectCircles(self.grayImg, 15, 15, [10, 60])
        self.displayCircles(res)
        self.ui.edgeView.setPixmap(QPixmap(r".\images\houghCircle.png"))
    def detectCircles(self,img, threshold, region, radius):
        # detectCircles takes img array,threshold of min points,region and radius range param
        M, N = img.shape

        [minR, maxR] = radius

        R = maxR - minR
        # Acc. array for the radius, X and Y
        # Also appending a padding of 2 times maxR.
        A = np.zeros((maxR, M + 2 * maxR, N + 2 * maxR))
        B = np.zeros((maxR, M + 2 * maxR, N + 2 * maxR))

        # making angles array
        angles = np.arange(0, 360) * np.pi / 180
        # Extracting all edge indices from filtered img
        edges = np.argwhere(img[:, :])
        for v in range(R):
            r = minR + v
            circleBprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
            (m, n) = (r + 1, r + 1)  # the center of the blueprint
            for angle in angles:
                x = int(np.round(r * np.cos(angle)))
                y = int(np.round(r * np.sin(angle)))
                circleBprint[m + x, n + y] = 1
            const = np.argwhere(circleBprint).shape[0]
            for x, y in edges:
                # For each edge coordinates
                # Centering the blueprint circle over the edges
                # and updating the accumulator array
                X = [x - m + maxR, x + m + maxR]  # Computing the extreme X values
                Y = [y - n + maxR, y + n + maxR]  # Computing the extreme Y values
                A[r, X[0]:X[1], Y[0]:Y[1]] += circleBprint
            A[r][A[r] < threshold * const / r] = 0

        for r, x, y in np.argwhere(A):
            temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
            try:
                p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
            except:
                continue
            B[r + (p - region), x + (a - region), y + (b - region)] = 1

        return B[:, maxR:-maxR, maxR:-maxR]

    def displayCircles(self,A):
        fig = plt.figure()
        plt.imshow(self.image)
        circleCoordinates = np.argwhere(A)  # Extracting the circle information
        circle = []
        for r, x, y in circleCoordinates:
            circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
            fig.add_subplot(111).add_artist(circle[-1])
        plt.axis('off')
        plt.savefig('houghCircle.png', bbox_inches='tight', pad_inches=0)

    def rgb2gray(self, rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

    def gaussianFilter(self):
        n = 5
        sigma = 1.4
        kernel = self.gaussian_kernel(n, sigma)
        img_smoothed = convolve(self.grayImg, kernel)
        return img_smoothed

    def gaussian_kernel(self,size, sigma):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    def highFilter(self,img):
        maskX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        maskY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        n = 3
        R, C = img.shape
        s1values = np.zeros((R + n - 1, C + n - 1))
        s2values = np.zeros((R + n - 1, C + n - 1))
        newImage = np.zeros((R + n - 1, C + n - 1))
        for i in range(1, R - 2):
            for j in range(1, C - 2):
                S1 = np.sum(np.multiply(maskX, img[i:i + n, j:j + n]))
                S2 = np.sum(np.multiply(maskY, img[i:i + n, j:j + n]))
                s1values[i + 1, j + 1] = S1
                s2values[i + 1, j + 1] = S2
                newImage[i, j] = np.sqrt(np.power(S1, 2) + np.power(S2, 2))

        angles = np.arctan2(s2values, s1values)
        newImage *= 255.0 / newImage.max()
        return newImage, angles

    def non_max_suppression(self,img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass
        return Z

    def threshold(self,img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res, weak, strong

    def hysteresis(self,img, weak, strong=255):
        M, N = img.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                        img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
