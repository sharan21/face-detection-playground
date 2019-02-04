import numpy as np
import cv2 as cv
import tqdm as tq


cascade_path = './haarcascade_frontalface_default.xml'
path = 'sample_pictures/group1.jpg'

def detectFace(cascade_path, path):

    img = cv.imread(path)
    face_cascade = cv.CascadeClassifier(cascade_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    crop_img = []
    gray_img = []

    faces = face_cascade.detectMultiScale(gray, 1.5, 1)
    for (x,y,w,h) in faces:
        #  cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        crop_img.append(img[y:y+h, x:x+w])
        gray_img.append(img[y:y + h, x:x + w])

    print("writing cropped images into directories")

    for i in range(0, len(crop_img), 1):
        filename = "cropped_images/img{}.jpg".format(i)
        filenamegray = "grayimg{}.jpg".format(i)
        cv.imwrite(filename, crop_img[i])

    print("done saving images")

# MAIN

detectFace(cascade_path, path)

# cv.imshow('img',crop_img)
# cv.waitKey(0)
# cv.destroyAllWindows()










