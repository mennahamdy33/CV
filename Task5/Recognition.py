from tkinter import Toplevel, Label
import matplotlib.pyplot as plt #plot import
import matplotlib.colors  #color import
import numpy as np  #importing numpy
from PIL import Image #importing PIL to read all kind of images
from PIL import ImageTk
import glob
import cv2


def reading_faces_and_displaying():
    face_array = []
    for face_images in glob.glob('G:/SBME/CV/Tasks/PCA-and-Eigen-Faces/Eigenfaces/Train/*.jpg'): # assuming jpg
        face_image=Image.open(face_images)
        face_image = np.asarray(face_image,dtype=float)/255.0
        face_array.append(face_image)
    face_array=np.asarray(face_array)
    return face_array
def performing_pca(face_array):
    mean = np.mean(face_array, 0)
    flatten_Array = []
    for x in range(len(face_array)):
        flat_Array = face_array[x].flatten()
        flatten_Array.append(flat_Array)
    flatten_Array = np.asarray(flatten_Array)
    mean = mean.flatten()
    # flatten_Array=flatten_Array.T
    #print(flatten_Array.shape)
    #face_array = face_array.flatten()
    # mean=mean.T
    #substract_mean_from_original = np.subtract(flatten_Array, mean)
    # transpose_substract_mean_from_original=substract_mean_from_original.T
    # eigen_faces=displaying_eigen_faces(face_array,mean)
    #covariance_matrix = np.cov(substract_mean_from_original)
    #eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    return mean,flatten_Array,  

def Eigen():
    face_array=reading_faces_and_displaying()
    mean,flatten_Array=performing_pca(face_array) # eigen_values,eigen_vectors
    substract_mean_from_original = np.subtract(flatten_Array, mean)
    U, s, V = np.linalg.svd(substract_mean_from_original, full_matrices=False)
    print("marwa khody balek",len(V))
    Eigen_faces=[]
    for x in range(V.shape[0]):
        fig=np.reshape(V[x],(425,425))
        Eigen_faces.append(fig)
    return Eigen_faces,face_array,mean,substract_mean_from_original,V

def class_face(k,test_from_mean,test_flat_images,V,substract_mean_from_original,face_array):
    eigen_weights = np.dot(V[:k, :],substract_mean_from_original.T)
    threshold = 6000
    for i in range(test_from_mean.shape[0]):
        test_weight = np.dot(V[:k, :],test_from_mean[i:i + 1,:].T)
        distances_euclidian = np.sum((eigen_weights - test_weight) ** 2, axis=0)
        image_closest = np.argmin(np.sqrt(distances_euclidian))
        to_plot=np.reshape(test_flat_images[i,:], (425,425))
        cv2.imshow('to_plot', to_plot)
        cv2.waitKey(0)
        if (distances_euclidian[image_closest] <= threshold):
            cv2.imshow('face_array[image_closest,:,:]', face_array[image_closest,:,:])
            cv2.waitKey(0)
    plt.show()

def returning_vector(test_images):
    flat_test_Array = []
    for x in range(len(test_images)):
        flat_Array = test_images[x].flatten()
        flat_test_Array.append(flat_Array)
    flat_test_Array = np.asarray(flat_test_Array)
    return flat_test_Array

def reading_test_images():
    test_images=[]
    for images in glob.glob('G:/SBME/CV/Tasks/PCA-and-Eigen-Faces/Eigenfaces/Test/*.jpg'):  # assuming jpg
        test_faces = Image.open(images)
        test_faces = np.asarray(test_faces, dtype=float) / 255.0
        #test_faces = test_faces.convert('L')  #int
        #test_faces = np.asarray(test_faces, dtype=float) / 255.0  # Normalize the image to be between 0 to 1
        test=(425,425,3)
        if test_faces.shape == test:
            test_faces=test_faces[:,:,0]
            test_images.append(test_faces)
        else:
            test_images.append(test_faces)
    print(len(test_images))
    #print(test_images[25].shape)
    #print(test_images[25].shape[0])
    #if test_images[25].shape[0]:
    #    print("a")
    flat_test_Array=returning_vector(test_images)
    test_images=np.asarray(test_images)
    return flat_test_Array,test_images

def function():
    Eigen_faces,face_array,mean,substract_mean_from_original,V = Eigen()
    test_flat_images,test_images=reading_test_images()
    test_from_mean=np.subtract(test_flat_images,mean)

    k=15
    print("FACES FOR K=2")
    class_face(k,test_from_mean,test_flat_images,V,substract_mean_from_original,face_array)

function()