import numpy as np
import cv2 
import os


def detect_faces(test_image, scaleFactor = 1.1, minNeighbors = 1):
    image_copy = test_image.copy()
    haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
    haar_cascade_SideFace = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    faces_rect = haar_cascade_face.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 1)
    print("Faces Founds:" , len(faces_rect))
    if (len(faces_rect) != 0):
        for (x,y,w,h) in faces_rect:
            cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,0,0),5)
    
    elif (len(faces_rect)==0):
        SideFace = haar_cascade_SideFace.detectMultiScale(gray_image,scaleFactor = 1.1, minNeighbors = 1)
        print("Side faces Founds:" , len(SideFace))
        for (sx,sy,sw,sh) in SideFace:
            cv2.rectangle(image_copy,(sx,sy),(sx+sw,sy+sh),(255,255,255),5)
        
    return image_copy , len(faces_rect)


# directory = "training-originals"
# counter =0 
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") : 
#         path = os.path.join(directory, filename)
#         test_image2 = cv2.imread(path)
#         faces , count= detect_faces( test_image2)
#         if (count == 1):
#             counter = counter +1
#         cv2.imwrite("./test/{}".format(filename),faces)
#     else:
#         continue
# print(counter)