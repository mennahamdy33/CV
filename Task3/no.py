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

def match_template_corr( x , temp ):
    y = np.empty(x.shape)
    y = correlate2d(x,temp,'same')
    return y

def match_template_corr_zmean( x , temp ):
    return match_template_corr(x , temp - temp.mean())

def match_template_ssd( x , temp ):
    term1 = np.sum( np.square( temp ))
    term2 = -2*correlate2d(x, temp,'same')
    term3 = correlate2d( np.square( x ), np.ones(temp.shape),'same' )
    ssd = np.maximum( term1 + term2 + term3 , 0 )
    return 1 - np.sqrt(ssd)

def match_template_xcorr( f , t ):
    f_c = f - correlate2d( f , np.ones(t.shape)/np.prod(t.shape), 'same') 
    t_c = t - t.mean()
    numerator = correlate2d( f_c , t_c , 'same' )
    d1 = correlate2d( np.square(f_c) , np.ones(t.shape), 'same')
    d2 = np.sum( np.square( t_c ))
    denumerator = np.sqrt( np.maximum( d1 * d2 , 0 )) # to avoid sqrt of negative
    response = np.zeros( f.shape )
    valid = denumerator > np.finfo(np.float32).eps # mask to avoid division by zero
    response[valid] = numerator[valid]/denumerator[valid]
    return response
 

imgs_dir = r'D:\CV\task#3'
imgs_names = ['chess.png']
imgs_fnames = [ join( imgs_dir, img_name) for img_name in imgs_names ]
imgs_rgb = [ np.array(Image.open(img)) for img in imgs_fnames ]
imgs_gray = [ rgb2gray( img ) for img in imgs_rgb ]

print(imgs_gray[0])
template = np.array(Image.open(r'D:\CV\CV\Task3\images\temp_chess.png'))
templates = [rgb2gray(template)]
fig, ax = plt.subplots(1,2,figsize = (15, 10))
path1 = r"D:\CV\task#3\chess.png"
path2 = r"D:\CV\CV\Task3\images\temp_chess.png"
image = cv2.imread(path1,1)
h = cv2.imread(path2,0)


matches_ssd = [ match_template_ssd(x,h) for (x,h) in zip(imgs_gray,templates)]
matches_xcorr = [ match_template_xcorr(x,h) for (x,h) in zip(imgs_gray,templates)]

# matches_corr_maxima = [ nms.local_maxima(x,min(t.shape)//8) for (x,t) in zip(matches_corr,templates)]
# matches_corr_zmean_maxima = [ nms.local_maxima(x,min(t.shape)//8) for (x,t) in zip(matches_corr_zmean,templates)]
matches_ssd_maxima = [ nms.local_maxima(x,min(t.shape)) for (x,t) in zip(matches_ssd,templates)]
matches_xcorr_maxima = [ nms.local_maxima(x,min(t.shape)//8) for (x,t) in zip(matches_xcorr,templates)]

methods_n = 2
# patches = zip(imgs_gray,templates,matches_ssd,matches_xcorr,matches_ssd_maxima,matches_xcorr_maxima)
patches = zip(imgs_gray,templates,matches_ssd,matches_xcorr,matches_ssd_maxima,matches_xcorr_maxima )



fig, ax = plt.subplots(len(imgs_gray)*(methods_n+1),2,figsize = (20, 40))
plt.autoscale(True)
img = imgs_gray[0]

for i,(im,temp,mssd,mxcorr,pssd,pxcorr) in enumerate(patches):
    
    def get_rect_on_maximum(y,template):
        ij = np.unravel_index(np.argmax(y), y.shape)
        x, y = ij[::-1]
        # highlight matched region
        htemp, wtemp = temp.shape
        startX = int(x - wtemp/2)
        startY = int(y - htemp/2)
        endX = int(x + wtemp/2)
        endY = int(y + wtemp/2)
        print("get_rect_on_maximum")
        cv2.rectangle(image, (startX,startY),(endX, endY) , (0,255,255), 2)

    
    def make_rects(xy,template):
        htemp, wtemp = template.shape
        print("make_rects")
        for ridx in range(xy.shape[0]):
            y,x = xy[ridx]
            # r =  plt.Rectangle((x-wtemp/2, y-htemp/2), wtemp, htemp, edgecolor='g', facecolor='none')
            startX = int(x - wtemp/2)
            startY = int(y - htemp/2)
            endX = int(x + wtemp/2)
            endY = int(y + wtemp/2)
            cv2.rectangle(image, (startX,startY),(endX, endY) , (255,255,255), 2)

       
    
    print("for",i)
    
    get_rect_on_maximum( mssd ,temp)
   
    make_rects( pssd , temp )

    cv2.imwrite("D:\CV\CV\Task3\images\TestYarab1.jpeg", image)
    
    get_rect_on_maximum(mxcorr,temp)
   
    make_rects( pxcorr, temp )
  
    cv2.imwrite("D:\CV\CV\Task3\images\TestYarab.jpeg", image)



