import numpy as np
from os import listdir
from PIL import Image
import importlib
from scipy.signal import convolve2d
from scipy import signal,ndimage
from skimage.transform import resize,rescale,rotate
from math import sqrt,sin, cos
import itertools
import sys
import matplotlib.pyplot as plt
import random
import cv2
from matplotlib import cm
# import numpy as np
# import cv2
import glob
from scipy.ndimage.filters import convolve

class Sift:

    N_OCTAVES = 4 
    N_SCALES = 5 
    SIGMA = 1.6
    K = sqrt(2)
    
    def __init__(self , Image1 , Image2):
        # 
        self.img_rgb = np.array(Image.open(Image1))
        # res = cv2.resize(self, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
        self.img_rgb_used,ratio = self.sift_resize(self.img_rgb,0.6)
        self.imgs_gray = self.rgb2gray(self.img_rgb_used)
        self.img_sift = self.pipeline(self.imgs_gray)
        self.img2_rgb,_ = self.sift_resize(np.array(Image.open(Image2)), ratio )
        # self.img2_rgb = rotate(self.img2_rgb,90)
        self.imgs_gray2 = self.rgb2gray(self.img2_rgb)
        self.img_sift2 = self.pipeline(self.imgs_gray2)

    def sift_timing(self):
        if (self.img_sift2):
            return "done"

    def OutPut(self,flag):
        outputImage = self.match2(self.img_rgb_used, self.img_sift[0], self.img_sift[1], self.img2_rgb, self.img_sift2[0], self.img_sift2[1],flag)
        return (outputImage)

    def Kernal(self):
        SIGMA_SEQ = lambda s: [ (self.K**i)*s for i in range(self.N_SCALES) ]
        SIGMA_SIFT = SIGMA_SEQ(self.SIGMA) #
        KERNEL_RADIUS = lambda s : 2 * int(round(s))
        KERNELS_SIFT = [ self.gaussian_kernel2d(std = s, 
                                        kernlen = 2 * KERNEL_RADIUS(s) + 1) 
                        for s in SIGMA_SIFT ]
        return KERNELS_SIFT                 

    def rgb2gray(self,rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

    def gaussian_kernel1d(self,kernlen=7, std=1.5):
        kernel1d = signal.gaussian(kernlen, std=std)
        kernel1d = kernel1d.reshape(kernlen, 1)
        return kernel1d / kernel1d.sum()

    def gaussian_kernel2d(self,kernlen=7, std=1.5):
        gkern1d = self.gaussian_kernel1d(kernlen,std)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def padded_slice(self,img, sl):
        output_shape = np.asarray(np.shape(img))
        output_shape[0] = sl[1] - sl[0]
        output_shape[1] = sl[3] - sl[2]
        src = [max(sl[0], 0),
            min(sl[1], img.shape[0]),
            max(sl[2], 0),
            min(sl[3], img.shape[1])]
        dst = [src[0] - sl[0], src[1] - sl[0],
            src[2] - sl[2], src[3] - sl[2]]
        output = np.zeros(output_shape, dtype=img.dtype)
        output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
        return output

    def sift_resize(self,img, ratio = None):
        ratio = ratio if ratio is not None else np.sqrt((1024*1024) / np.prod(img.shape[:2]))
        newshape = list(map( lambda d : int(round(d*ratio)), img.shape[:2])) 
        img = resize( img, newshape , anti_aliasing = True )
        return img,ratio

    def sift_gradient(self,img):
        dx = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
        dy = dx.T
        gx = signal.convolve2d( img , dx , boundary='symm', mode='same' )
        gy = signal.convolve2d( img , dy , boundary='symm', mode='same' )
        magnitude = np.sqrt( gx * gx + gy * gy )
        direction = np.rad2deg( np.arctan2( gy , gx )) % 360
        return gx,gy,magnitude,direction   

    def image_dog(self, img ):
        octaves = []
        dog = []
        base = rescale( img, 2, anti_aliasing=False) 
        # R = self.HarrisCornerDetection(dog)
        kernal = self.Kernal()
        octaves.append([ convolve2d( base , kernel , 'same', 'symm') 
                            for kernel in kernal ])
        dog.append([ s2 - s1 
                for (s1,s2) in zip( octaves[0][:-1], octaves[0][1:])])
        for i in range(1,self.N_OCTAVES):
            base = octaves[i-1][2][::2,::2] # 2x subsampling 
            octaves.append([base] + 
                [convolve2d( base , kernel , 'same', 'symm') 
                            for kernel in kernal[1:] ])
            dog.append([ s2 - s1 
                for (s1,s2) in zip( octaves[i][:-1], octaves[i][1:])])
        return dog , octaves  

    def corners( self,dog , r = 10 ):
        threshold = ((r + 1.0)**2)/r
        dx = np.array([-1,1]).reshape((1,2))
        dy = dx.T
        dog_x = convolve2d( dog , dx , boundary='symm', mode='same' )
        dog_y = convolve2d( dog , dy , boundary='symm', mode='same' )
        dog_xx = convolve2d( dog_x , dx , boundary='symm', mode='same' )
        dog_yy = convolve2d( dog_y , dy , boundary='symm', mode='same' )
        dog_xy = convolve2d( dog_x , dy , boundary='symm', mode='same' )
        
        tr = dog_xx + dog_yy
        det = dog_xx * dog_yy - dog_xy ** 2
        response = ( tr**2 +10e-8) / (det+10e-8)
        
        coords = list(map( tuple , np.argwhere( response < threshold ).tolist() ))
        return coords  
    
    def contrast(self, dog , img_max, threshold = 0.03 ):
        dog_norm = dog / img_max
        coords = list(map( tuple , np.argwhere( np.abs( dog_norm ) > threshold ).tolist() ))
        return coords

    def cube_extrema(self, img1, img2, img3 ):
        value = img2[1,1]

        if value > 0:
            return all([np.all( value >= img ) for img in [img1,img2,img3]]) # test map
        else:
            return all([np.all( value <= img ) for img in [img1,img2,img3]]) # test map 

    def dog_keypoints(self, img_dogs , img_max , threshold = 0.03 ):
        octaves_keypoints = []
    
        for octave_idx in range(self.N_OCTAVES):
            img_octave_dogs = img_dogs[octave_idx]
            keypoints_per_octave = []
            for dog_idx in range(1, len(img_octave_dogs)-1):
                dog = img_octave_dogs[dog_idx]
                keypoints = np.full( dog.shape, False, dtype = np.bool)
                candidates = set( (i,j) for i in range(1, dog.shape[0] - 1) for j in range(1, dog.shape[1] - 1))
                search_size = len(candidates)
                candidates = candidates & set(self.corners(dog)) & set(self.contrast( dog , img_max, threshold ))
                search_size_filtered = len(candidates)
                for i,j in candidates:
                    slice1 = img_octave_dogs[dog_idx -1][i-1:i+2, j-1:j+2]
                    slice2 = img_octave_dogs[dog_idx   ][i-1:i+2, j-1:j+2]
                    slice3 = img_octave_dogs[dog_idx +1][i-1:i+2, j-1:j+2]
                    if self.cube_extrema( slice1, slice2, slice3 ):
                        keypoints[i,j] = True
                keypoints_per_octave.append(keypoints)
            octaves_keypoints.append(keypoints_per_octave)
        return octaves_keypoints

    def dog_keypoints_orientations(self, img_gaussians , keypoints , num_bins = 36 ):
        kps = []
        for octave_idx in range(self.N_OCTAVES):
            img_octave_gaussians = img_gaussians[octave_idx]
            octave_keypoints = keypoints[octave_idx]
            for idx,scale_keypoints in enumerate(octave_keypoints):
                scale_idx = idx + 1 ## idx+1 to be replaced by quadratic localization
                gaussian_img = img_octave_gaussians[ scale_idx ] 
                sigma = 1.5 * self.SIGMA * ( 2 ** octave_idx ) * ( self.K ** (scale_idx))
                
                
                KERNEL_RADIUS = lambda s : 2 * int(round(s))
                radius = KERNEL_RADIUS(sigma)
                kernel = self.gaussian_kernel2d(std = sigma, kernlen = 2 * radius + 1)
                gx,gy,magnitude,direction = self.sift_gradient(gaussian_img)
                direction_idx = np.round( direction * num_bins / 360 ).astype(int)          
                
                for i,j in map( tuple , np.argwhere( scale_keypoints ).tolist() ):
                    window = [i-radius, i+radius+1, j-radius, j+radius+1]
                    mag_win = self.padded_slice( magnitude , window )
                    dir_idx = self.padded_slice( direction_idx, window )
                    weight = mag_win * kernel 
                    hist = np.zeros(num_bins, dtype=np.float32)
                    
                    for bin_idx in range(num_bins):
                        hist[bin_idx] = np.sum( weight[ dir_idx == bin_idx ] )
                
                    for bin_idx in np.argwhere( hist >= 0.8 * hist.max() ).tolist():
                        angle = (bin_idx[0]+0.5) * (360./num_bins) % 360
                        kps.append( (i,j,octave_idx,scale_idx,angle))
        return kps  

    def rotated_subimage(self,image, center, theta, width, height):
        theta *= 3.14159 / 180 # convert to rad
        
        
        v_x = (cos(theta), sin(theta))
        v_y = (-sin(theta), cos(theta))
        s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
        s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

        mapping = np.array([[v_x[0],v_y[0], s_x],
                            [v_x[1],v_y[1], s_y]])

        return cv2.warpAffine(image,mapping,(width, height),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)
    
    def extract_sift_descriptors128(self, img_gaussians, keypoints, num_bins = 8 ):
        descriptors = []; points = [];  data = {} # 
        for (i,j,oct_idx,scale_idx, orientation) in keypoints:

            if 'index' not in data or data['index'] != (oct_idx,scale_idx):
                data['index'] = (oct_idx,scale_idx)
                gaussian_img = img_gaussians[oct_idx][ scale_idx ] 
                sigma = 1.5 * self.SIGMA * ( 2 ** oct_idx ) * ( self.K ** (scale_idx))
                data['kernel'] = self.gaussian_kernel2d(std = sigma, kernlen = 16)                

                gx,gy,magnitude,direction = self.sift_gradient(gaussian_img)
                data['magnitude'] = magnitude
                data['direction'] = direction

            window_mag = self.rotated_subimage(data['magnitude'],(j,i), orientation, 16,16)
            window_mag = window_mag * data['kernel']
            window_dir = self.rotated_subimage(data['direction'],(j,i), orientation, 16,16)
            window_dir = (((window_dir - orientation) % 360) * num_bins / 360.).astype(int)

            features = []
            for sub_i in range(4):
                for sub_j in range(4):
                    sub_weights = window_mag[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                    sub_dir_idx = window_dir[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                    hist = np.zeros(num_bins, dtype=np.float32)
                    for bin_idx in range(num_bins):
                        hist[bin_idx] = np.sum( sub_weights[ sub_dir_idx == bin_idx ] )
                    features.extend( hist.tolist())
            features = np.array(features) 
            features /= (np.linalg.norm(features))
            np.clip( features , np.finfo(np.float16).eps , 0.2 , out = features )
            assert features.shape[0] == 128, "features missing!"
            features /= (np.linalg.norm(features))
            descriptors.append(features)
            points.append( (i ,j , oct_idx, scale_idx, orientation))
        return points , descriptors  

    def pipeline(self, input_img ):
        img_max = input_img.max()
        dogs, octaves = self.image_dog( input_img )
        keypoints = self.dog_keypoints( dogs , img_max , 0.03 )
        keypoints_ijso = self.dog_keypoints_orientations( octaves , keypoints , 36 )
        points,descriptors = self.extract_sift_descriptors128(octaves , keypoints_ijso , 8)
        return points, descriptors
    
    def kp_list_2_opencv_kp_list(self,kp_list):
        opencv_kp_list = []
        for kp in kp_list:
            opencv_kp = cv2.KeyPoint(x=kp[1] * (2**(kp[2]-1)),
                                        y=kp[0] * (2**(kp[2]-1)),
                                        _size=kp[3],
                                        _angle=kp[4]
                                    )
            opencv_kp_list += [opencv_kp]

        return opencv_kp_list    

    def match2(self, img_a, pts_a, desc1, img_b, pts_b, desc2,flag):
        img_a, img_b = tuple(map( lambda i: np.uint8(i*255), [img_a,img_b] ))
    
        desc1 = np.array(desc1)
        desc2 = np.array(desc2)
        
        assert desc1.ndim == 2

        assert desc2.ndim == 2
        
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        numKeyPoints1 = desc1.shape[0]
        numKeyPoints2 = desc2.shape[0]
        matches = []
        good = []
        pts_a = self.kp_list_2_opencv_kp_list(pts_a)
        pts_b = self.kp_list_2_opencv_kp_list(pts_b)
        i = 0


        if (flag == 0 ):
            matches = []
            good = []
            for x in range(numKeyPoints1):
                distance = -1
                y_ind = -1
                for y in range(numKeyPoints2):
                    sumSquare = 0
                    for m in range(desc1.shape[1]):
                        sumSquare += (desc1[x][m] - desc2[y][m]) **2
                    sumSquare = np.sqrt(sumSquare)
                    if distance < 0 or (sumSquare < distance and distance >=0):
                        distance = sumSquare
                        y_ind = y
            
                cur = cv2.DMatch()
                cur.queryIdx = x
                cur.trainIdx = y_ind
                cur.distance = distance
                matches.append(cur)
                if (cur.distance < 0.23 ):
                    good.append(cur)
                    i+=1
        
        elif(flag == 1):
            matches = []
            good = []
            for x in range (numKeyPoints1):
                distance = -1
                y_ind = -1
                mean_desc1 = np.mean(desc1[x][:])
                std_desc1 = np.std(desc1[x][:])
            

                for y in range(numKeyPoints2):
                
                    ncorr = 0
                    mean_desc2 = np.mean(desc2[y][:])
                    std_desc2 = np.std(desc2[y][:])
                
                    result = np.mean(np.multiply((desc1[x][:]-mean_desc1),(desc2[y][:] - mean_desc2)))
        
                    ncorr = (result/(std_desc1*std_desc2))
                        
                    if distance < 0 or (ncorr > distance and distance >=0):
                        distance = ncorr
                        y_ind = y
            
                cur = cv2.DMatch()
                cur.queryIdx = x
                cur.trainIdx = y_ind
                cur.distance = distance
                matches.append(cur)
                
                if (cur.distance > 0.97 ):
                    good.append(cur)
                    i+=1

        output = self.drawMatches(img_a, pts_a, img_b, pts_b, good )
        cv2.imwrite("./images/M"+str(flag)+".png",  cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        return output
  
    def concatImages(self, imgs):
        # Skip Nones
        imgs = [img for img in imgs if img is not None]
        maxh = max([img.shape[0] for img in imgs]) if imgs else 0
        sumw = sum([img.shape[1] for img in imgs]) if imgs else 0
        vis = np.zeros((maxh, sumw, 3), np.uint8)
        vis.fill(255)
        accumw = 0
        for img in imgs:
            h, w = img.shape[:2]
            vis[:h, accumw:accumw+w, :] = img
            accumw += w

        return vis

    def drawMatches(self, img1, kp1, img2, kp2, matches ):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
     
        vis = self.concatImages([img1, img2])
        kp_pairs = [[kp1[m.queryIdx], kp2[m.trainIdx]] for m in matches]
        status = np.ones(len(kp_pairs), np.bool_)
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.circle(vis, (x1, y1), 2, green, 1)
                cv2.circle(vis, (x2, y2), 2, green, 1)
            else:
                r = 2
                thickness = 1
                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), white, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), white, thickness)
                cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), white, thickness)
                cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), white, thickness)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), white , 1)

        return vis      

    # def HarrisCornerDetection(self,image):
    
    #     # The two Sobel operators - for x and y direction
    #     SobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #     SobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    #     w, h = image.shape

    #     # X and Y derivative of image using Sobel operator
    #     ImgX = cv2.Sobel(image, cv2.CV_64F,1 , 0, ksize=1)
    #     ImgY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

    #     # # Eliminate the negative values
    #     # There are multiple ways this can be done
    #     # 1. Off setting with a positive value (commented out below)
    #     # 2. Setting negative values to Zero (commented out)
    #     # 3. Multiply by -1 (implemented below, found most reliable method)
    #     # ImgX += 128.0
    #     # ImgY += 128.0
    #     for ind1 in range(w):
    #         for ind2 in range(h):
    #             if ImgY[ind1][ind2] < 0:
    #                 ImgY[ind1][ind2] *= -1
    #                 # ImgY[ind1][ind2] = 0
    #             if ImgX[ind1][ind2] < 0:
    #                 ImgX[ind1][ind2] *= -1
    #                 # ImgX[ind1][ind2] = 0

    #     # Display the output results after Sobel operations
    #     # cv2.imshow("SobelX", ImgX)
    #     # cv2.imshow("SobelY", ImgY)

    #     ImgX_2 = np.square(ImgX)
    #     ImgY_2 = np.square(ImgY)

    #     ImgXY = np.multiply(ImgX, ImgY)
    #     ImgYX = np.multiply(ImgY, ImgX)

    #     #Use Gaussian Blur

    #     ImgX_2 =  cv2.GaussianBlur(ImgX_2,(5,5),0)
    #     ImgY_2 = cv2.GaussianBlur(ImgY_2,(5,5),0)
    #     ImgXY = cv2.GaussianBlur(ImgXY,(5,5),0)
    #     ImgYX = cv2.GaussianBlur(ImgYX,(5,5),0)

    #     alpha = 0.06
    #     R = np.zeros((w, h), np.float32)
    #     # For every pixel find the corner strength
    #     for row in range(w):
    #         for col in range(h):
    #             M_bar = np.array([[ImgX_2[row][col], ImgXY[row][col]], [ImgYX[row][col], ImgY_2[row][col]]])
    #             R[row][col] = np.linalg.det(M_bar) - (alpha * np.square(np.trace(M_bar)))
    #     return R

    # def GetSobel(self,image, Sobel, width, height):
    #     # Initialize the matrix
    #     I_d = np.zeros((width, height), np.float32)

    #     # For every pixel in the image
    #     for rows in range(width):
    #         for cols in range(height):
    #             # Run the Sobel kernel for each pixel
    #             if rows >= 1 or rows <= width-2 and cols >= 1 or cols <= height-2:
    #                 for ind in range(3):
    #                     for ite in range(3):
    #                         I_d[rows][cols] += Sobel[ind][ite] * image[rows - ind - 1][cols - ite - 1]
    #             else:
    #                 I_d[rows][cols] = image[rows][cols]

    #     return I_d

    # def gaussianFilter(self,grayImg):
    #     n = 5
    #     sigma = 1.4
    #     kernel = self.gaussian_kernel(n, sigma)
    #     img_smoothed = convolve(grayImg, kernel)
    #     return img_smoothed

    # def gaussian_kernel(self, size, sigma):
    #     size = int(size) // 2
    #     x, y = np.mgrid[-size:size + 1, -size:size + 1]
    #     normal = 1 / (2.0 * np.pi * sigma ** 2)
    #     g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    #     return g

    # def corners(self, img):
    #     # firstimage = cv2.imread("./images/cow.png")
    #     # greyimg = self.rgb2gray(img)
    #     w, h = img.shape
    #     # Corner detection
    #     R = self.HarrisCornerDetection(img)
    #     ave = R.mean()

    #     # Empirical Parameter
    #     # This parameter will need tuning based on the use-case
    #     CornerStrengthThreshold = 2.18099991e-08

    #     # Plot detected corners on image
    #     radius = 2
    #     color = (0, 255, 0)  # Green
    #     thickness = 1

    #     PointList = []
    #     # Look for Corner strengths above the threshold
    #     for row in range(w):
    #         for col in range(h):
    #             if R[row][col] > CornerStrengthThreshold:
    #                 max = R[row][col]

    #                 # Local non-maxima suppression
    #                 skip = False
    #                 for nrow in range(5):
    #                     for ncol in range(5):
    #                         if row + nrow - 2 < w and col + ncol - 2 < h:
    #                             if R[row + nrow - 2][col + ncol - 2] > max:
    #                                 skip = True
    #                                 break

    #                 if not skip:
    #                     # Point is expressed in x, y which is col, row
    #                     # cv2.circle(img, (col, row), radius, color, thickness)
    #                     PointList.append((row, col))

    #     # Display image indicating corners and save it
    #     return PointList    


    ## DH MATCH EL FUNCTION

    # def match(self, img_a, pts_a, desc_a, img_b, pts_b, desc_b):
    #     img_a, img_b = tuple(map( lambda i: np.uint8(i*255), [img_a,img_b] ))
        
    #     desc_a = np.array( desc_a , dtype = np.float32 )
    #     desc_b = np.array( desc_b , dtype = np.float32 )

    #     pts_a = self.kp_list_2_opencv_kp_list(pts_a)
    #     pts_b = self.kp_list_2_opencv_kp_list(pts_b)

    #     # create BFMatcher object
    #     # BFMatcher with default params
    #     bf = cv2.BFMatcher()
    #     matches = bf.knnMatch(desc_a,desc_b,k=2)
    #     # Apply ratio test
    #     good = []
    #     for m,n in matches:
    #         if m.distance < 0.25*n.distance:
    #             good.append(m)

    #     img_match = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)

    #     cv2.drawMatches(img_a,pts_a,img_b,pts_b,good, outImg = img_match,
    #                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                  
    #     return img_match
