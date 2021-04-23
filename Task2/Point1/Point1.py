import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy.ndimage.filters import convolve

path = "./final.png"

# menna add canny code heree

img = cv2.imread(path, 0)


def detectCircles(img, threshold, region, radius):
    # detectCircles takes img array,threshold of min points,region and radius range param
    M, N = img.shape

    [minR, maxR] = radius

    R = maxR - minR
    # Acc. array for the radius, X and Y
    # Also appending a padding of 2 times maxR to overcome the problems of overflow!!
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


def displayCircles(A):
    img = cv2.imread(path)
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)  # Extracting the circle information
    circle = []
    for r, x, y in circleCoordinates:
        circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.show()


res = detectCircles(img, 15, 15, [10, 60])
displayCircles(res)
