# import the necessary packages
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


def maximum_filter_ignore_nan(array, *args, **kwargs):
    nans = np.isnan(array)
    replaced = np.where(nans, -np.inf, array)
    return filters.maximum_filter(replaced, *args, **kwargs)


def local_maxima( x , size, threshold = 0.5 ):
    x_max = maximum_filter_ignore_nan(x,size)
    maxima = (x_max) >= (np.amax( x_max) * threshold)
    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(x, labeled, range(1, num_objects+1)))
    return xy

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
 
