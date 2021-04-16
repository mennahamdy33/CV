import cv2
import numpy as np
from scipy.ndimage.filters import convolve

path = "./1.jpg"


def rgb2gray(rgb_image):
    return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])


def show(image):
    cv2.imshow('sample image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def gaussianFilter(img):
    n = 5
    sigma = 1.4
    # R, C = img.shape
    kernel = gaussian_kernel(n, sigma)
    img_smoothed = convolve(img, kernel)
    # show(img_smoothed)

    # cv2.imwrite("./GaussianFilter.png", img_smoothed)
    return img_smoothed

def highFilter(img,file):
    maskX= np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    maskY= np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    n= 3
    R,C = img.shape
    s1values = np.zeros((R+n-1,C+n-1))
    s2values = np.zeros((R+n-1,C+n-1))
    newImage = np.zeros((R+n-1,C+n-1))
    for i in range(1,R-2):
        for j in range(1,C-2):
            S1 = np.sum(np.multiply(maskX,img[i:i+n,j:j+n]))
            S2 = np.sum(np.multiply(maskY,img[i:i+n,j:j+n]))
            s1values[i+1,j+1] = S1
            s2values[i+1,j+1] = S2
            newImage[i,j] = np.sqrt(np.power(S1,2)+np.power(S2,2))

    angles = np.arctan2(s2values,s1values)
    newImage *= 255.0 / newImage.max()
    cv2.imwrite("./oldSobel.png",newImage)
    return newImage,angles
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Ix = convolution(img, Kx, False)
    Ix = convolve(img, Kx)

    Iy = convolve(img, Ky)

    hyp = (Ix * Ix) + (Iy * Iy)
    G = np.sqrt(hyp, dtype=np.float32)
    G *= 255.0 / G.max()
    # gradient = G.astype(np.float32)
    theta = np.arctan2(Iy, Ix)
    # show(G)

    return G, theta


def non_max_suppression(img, D):
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
    # show(np.uint8(Z))
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
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


def hysteresis(img, weak, strong=255):
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


img = cv2.imread(path, 0)
# rgbImg = cv2.imread(path)
# grayImg = rgb2gray(rgbImg)
# show(img)
smoothedImg = gaussianFilter(img)
# filteredImg = cv2.imread("./GaussianFilter.png",0)
mg, angle = highFilter(smoothedImg, " ")

nonMaxImg = non_max_suppression(mg, angle)
thresh, weak, strong = threshold(nonMaxImg, 0.05, 0.09)
img_final = hysteresis(thresh, weak, strong)
cv2.imwrite("./final.png", img_final)
