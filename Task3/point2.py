import numpy as np
from scipy.signal import correlate2d
import itk
from cvutils import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import isfile , join
from PIL import Image
from skimage import io
from skimage.color import rgb2gray

import cv2

from itkwidgets import view
import itkwidgets
from IPython.display import display
import nms
import importlib
importlib.reload(nms)


def ssd(self,img,template):
    matches_ssd = nms.match_template_ssd((img), temp)
    matches_ssd_maxima = nms.local_maxima(x,min(t.shape))
    return(matches_ssd,matches_ssd_maxima)

def xcorr(self,img,template):
    matches_xcorr = nms.match_template_xcorr(img,template)
    matches_xcorr_maxima = nms.local_maxima(x,min(t.shape)//8)
    return(matches_xcorr,matches_xcorr_maxima)


imgs_dir = r'D:\CV\CV\Task3\images'
imgs_names = ['cell.jpeg']
imgs_fnames = [ join( imgs_dir, img_name) for img_name in imgs_names ]
imgs_rgb = np.array(Image.open(img)) 
imgs_gray = rgb2gray( img ) 

methods_n = 2

fig, ax = plt.subplots(len(imgs_gray)*(methods_n+1),2,figsize = (20, 40))
plt.autoscale(True)

for i,(im,temp,mssd,mxcorr,pssd,pxcorr) in enumerate(patches):
    
    def get_rect_on_maximum(y,template):
        ij = np.unravel_index(np.argmax(y), y.shape)
        x, y = ij[::-1]
        # highlight matched region
        htemp, wtemp = template.shape
        rect = plt.Rectangle((x-wtemp/2, y-htemp/2), wtemp, htemp, edgecolor='r', facecolor='none')
        return rect,x,y
    
    def make_rects(plt_object,xy,template):
        htemp, wtemp = template.shape
        for ridx in range(xy.shape[0]):
            y,x = xy[ridx]
            r =  plt.Rectangle((x-wtemp/2, y-htemp/2), wtemp, htemp, edgecolor='g', facecolor='none')
            cv2.rectangle(img, (x-wtemp/2, y-htemp/2) , (0,0,255), 2)

            plt_object.add_patch(r)
    
    def make_circles(plt_object,xy,template):
        htemp, wtemp = template.shape
        for ridx in range(xy.shape[0]):
            y,x = xy[ridx]
            plt_object.plot(x, y, 'o', markeredgecolor='g', markerfacecolor='none', markersize=10)
            
    row = (methods_n+1)*i 
    ax[row,0].imshow(im, cmap = 'gray')
    ax[row,1].imshow(temp, cmap = 'gray')

    r,x,y = get_rect_on_maximum(mssd,temp)
    ax[row + 1,0].imshow(im, cmap = 'gray')
    make_rects( ax[row + 1,0] , pssd, temp )
    ax[row + 1,0].add_patch(r)
    
    r,x,y = get_rect_on_maximum(mxcorr,temp)
    ax[row + 2,0].imshow(im, cmap = 'gray')
    make_rects( ax[row + 2,0] , pxcorr, temp )
    ax[row + 2,0].add_patch(r)
    ax[row + 2,1].imshow(mxcorr, cmap = 'gray')
    make_circles(ax[row + 2,1], pxcorr,temp)
    ax[row + 2,1].plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()

