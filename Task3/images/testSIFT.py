import numpy as np
import time
import cv2
def GetSobel(image, Sobel, width, height):
    I_d = np.zeros((width, height), np.float32)
    for rows in range(width):
        for cols in range(height):
            if rows >= 1 or rows <= width - 2 and cols >= 1 or cols <= height - 2:
                for ind in range(3):
                    for ite in range(3):
                        I_d[rows][cols] += Sobel[ind][ite] * image[rows - ind - 1][cols - ite - 1]
            else:
                I_d[rows][cols] = image[rows][cols]

    return I_d


# Descriptor Generation
def SIFT_descriptor(R, img):
    SIFT_desc_time_start = time.time()

    descriptors = []
    allKeyPoints = []

    sigma = 1.6  # from paper
    sigma = sigma * 1.5
    kernalWeidth = int(2 * np.ceil(sigma) + 1)  # 7

    SobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    SobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    num_subregion = 4
    num_bin = 8
    num_bins = 36
    bin_width = 360 // num_bins

    for keyPoint in R:
        x, y = int(keyPoint[0]), int(keyPoint[1])
        histogram = np.zeros(num_bins, dtype=np.float32)
        # 7x7 window around the key point
        sub_img = np.array(img[x - 3:x + 4, y - 3:y + 4])

        if (sub_img.shape[0] != 7 or sub_img.shape[1] != 7):
            continue
        else:
            width, height = sub_img.shape
            ImgX = GetSobel(sub_img, SobelX, width, height)
            ImgY = GetSobel(sub_img, SobelY, width, height)
            magnitude = np.sqrt(ImgX * ImgX + ImgY * ImgY)
            weighted_mag = cv2.GaussianBlur(magnitude, (kernalWeidth, kernalWeidth), sigma)
            direction = np.rad2deg(np.arctan2(ImgY, ImgX)) % 360

            # calculating the  histogram
            for i in range(len(direction)):
                for j in range(len(magnitude)):
                    direction_idx = int(np.floor(direction[i][j]) // bin_width)
                    histogram[direction_idx] += weighted_mag[i][j]

            # find dominant direction
            max_dir = np.argmax(histogram)
            max_mag = histogram[max_dir]
            angle = (max_dir + 0.5) * (360. / num_bins) % 360
            allKeyPoints.append([keyPoint[0], keyPoint[1], max_mag, angle])
            # finding new key points (value > 0.8 * max_val)
            for Dir, mag in enumerate(histogram):
                if Dir == max_dir:  # to avoid repetiton
                    continue
                if .8 * max_mag <= mag:
                    angle = (max_dir + 0.5) * (360. / num_bins) % 360
                    allKeyPoints.append([keyPoint[0], keyPoint[1], mag, angle])

    sigma = 1.5

    for keyPoint in allKeyPoints:
        x, y, win_mag, win_dir = int(keyPoint[0]), int(keyPoint[1]), keyPoint[2], keyPoint[3]
        # 15 X 15 window around the key point since cv2.GaussianBlur doesn't accept even masks
        sub_img = img[x - 8:x + 7, y - 8:y + 7]

        if (sub_img.shape[0] != 15 or sub_img.shape[1] != 15):
            continue
        else:
            width, height = sub_img.shape
            ImgX = GetSobel(sub_img, SobelX, width, height)
            ImgY = GetSobel(sub_img, SobelY, width, height)
            mag = np.sqrt(ImgX * ImgX + ImgY * ImgY)
            Dir = np.rad2deg(np.arctan2(ImgY, ImgX)) % 360
            weighted_mag = cv2.GaussianBlur(mag, (15, 15), sigma)

            # subtract the dominant direction
            Dir = (((Dir - win_dir) % 360) * num_bin / 360.).astype(int)
            features = []
            for sub_i in range(num_subregion):
                for sub_j in range(num_subregion):
                    sub_weights = weighted_mag[sub_i * 4:(sub_i + 1) * 4, sub_j * 4:(sub_j + 1) * 4]
                    sub_dir_idx = Dir[sub_i * 4:(sub_i + 1) * 4, sub_j * 4:(sub_j + 1) * 4]
                    histogram = np.zeros(num_bin, dtype=np.float32)
                    for bin_idx in range(num_bin):
                        histogram[bin_idx] = np.sum(sub_weights[sub_dir_idx == bin_idx])
                    features.extend(histogram.tolist())
            features /= (np.linalg.norm(np.array(features)))
            features = np.clip(features, np.finfo(np.float16).eps, 0.2)
            features /= (np.linalg.norm(features))
            descriptors.append(features)
    SIFT_desc_time_end = time.time()

    return allKeyPoints, descriptors, SIFT_desc_time_end - SIFT_desc_time_start