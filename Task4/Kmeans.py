import math 
from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random
import math 
from collections import defaultdict
import operator

class Kmeans:
    
    def __init__(self):
         
        self.res = None
        self.grey_l = [40,80,120,160,200,240]
           
    def Kmeans_Gray(self,grayImage):
        rows,columns = grayImage.shape
        self.histogram,self.bins = np.histogram(grayImage,256,[0,256])
        self.res = self.find_centroids_grayImage(self.histogram)
        end = np.zeros((grayImage.shape))
        
        if len(self.res[1]) > len(self.res[0]):
            '''bacground is res1'''
            flag = 1
        else:
            flag = 0


        for i in range(rows):
            for j in range(columns):
                if flag == 1:
                    if (grayImage[i][j] in self.res[1]):
                        end[i][j] = int(0)

                    else:
                        end[i][j] = int(1)
                else:
                    if (grayImage[i][j] in self.res[1]):
                        end[i][j] = int(1)

                    else:
                        end[i][j] = int(0)
        return(end)
        
    def find_centroids_grayImage(self,histogram):
        rand_points = [ random.randint(0, 255) for i in range(2) ]
        centroid1_avg = 0
        centroid2_avg = 0   
        for k in range(0,10):
            if k == 0:
                cent1, cent2 = rand_points
                
            else:
                cent1 = centroid1_avg
                cent2 = centroid2_avg

            point1_centroid = []
            point2_centroid = []
            w1_centroid = []
            w2_centroid = []
            sum1 = 0
            sum2 = 0
            for i,val in enumerate(histogram):
                ''' computing absolute distance from each of the cluster and assigning it to a particular cluster based on distance'''
                if  abs(i - cent1) <  abs(i - cent2):
                    point1_centroid.append(i)
                    w1_centroid.append(val)
                    sum1 = sum1 + (i * val)
                else:
                    point2_centroid.append(i)
                    w2_centroid.append(val)
                    sum2 = sum2 + (i * val)
            
            
            centroid1_avg = int(sum1)/sum(w1_centroid)	
            centroid2_avg = int(sum2)/sum(w2_centroid)			
        return [point1_centroid,point2_centroid] 
    
    def find_centroids_RGB(self,g):
        red_cent_list = []
        blue_cent_list = []
        green_cent_list = []
        zavg=[0,0,0]
        for i in range(0,6):
            array = np.matrix(g[i])
            avg = np.mean(array,0)
            pavg = np.ravel(avg)

            if not len(pavg):
                red_cent_list.append(zavg[0]) 
                blue_cent_list.append(zavg[1]) 
                green_cent_list.append(zavg[2])
            else:
                red_cent_list.append(pavg[0]) 
                blue_cent_list.append(pavg[1]) 
                green_cent_list.append(pavg[2])
        return[red_cent_list,blue_cent_list,green_cent_list] 
        
    def Kmeans_Color(self,Image):
        rows,columns = Image.shape[:2]
        r_points = [ random.randint(0, 255) for i in range(6) ]
        g_points = [ random.randint(0, 255) for i in range(6) ]
        b_points = [ random.randint(0, 255) for i in range(6) ]
        end = np.zeros((rows,columns))
     
        for it in range(0,10):    
            g = defaultdict(list)
            for r in range(rows):
                for c in range(columns):
                 
                    red, green, blue = Image[r][c][:3]
                   
                    distance_list = []

                    for k in range(0,6):
                        distance = math.sqrt(((int(r_points[k])- red)**2)+((int(g_points[k]) - green)**2)+((int(b_points[k])-blue)**2))
                        distance_list.append(distance)

                    index, value = min(enumerate(distance_list), key=operator.itemgetter(1))
                
                    end[r][c] = self.grey_l[index]

                    g[index].append([red,blue,green])

            r_points , b_points , g_points = self.find_centroids_RGB(g)
            return(end)
       
    def convertColorIntoGray(self,output):
        rows,columns = output.shape
        result = np.zeros((rows,columns))
        ref_val = output[0][0]
        for i in range(rows):
            for j in range(columns):
                if output[i][j] ==  ref_val:
                    result[i][j] = 1

                else:
                    result[i][j] = 0
        return result

