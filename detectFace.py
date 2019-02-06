import numpy as np
import cv2 as cv


class faceDetect:

    cascade_path = 'haarcascade_frontalface_default.xml'
    path = 'sample_pictures/group1.jpg'
    inputwidth = 25;
    inputheight = 25;


    def __init__(self, instance):
        self.instance = instance
        print("object initialised")

    def detectFaceOfImage(self,cascade_path, path):

        img = cv.imread(path)
        face_cascade = cv.CascadeClassifier(cascade_path)
        crop_img = []
        gray_img = []
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.5, 1)
        for (x, y, w, h) in faces:
            # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            crop_img.append(img[y:y + h, x:x + w])
            gray_img.append(img[y:y + h, x:x + w])



        # now we have an array of all the cropped faces in crop_img and gray_img, which need to be fixed for size
        # c
        self.checkAndFixSize(crop_img)

        print("writing cropped images into directories")

        for i in range(0, len(crop_img), 1):
            filename = "cropped_images/img{}.jpg".format(i)
            filenamegray = "cropped_gray_images/grayimg{}.jpg".format(i)
            cv.imwrite(filenamegray, crop_img[i])
            cv.imwrite(filename, gray_img[i])

        print("done saving images")

    def checkAndFixSize(self, crop_img):

        print("printing sizes of all images")

        for i in range(len(crop_img)):
            print("shape of the {}th image is".format(i+1))
            print(crop_img[i].shape)
            height, width, depth = crop_img[i].shape
            imgScale = self.inputwidth / width
            newX, newY = crop_img[i].shape[1] * imgScale, crop_img[i].shape[0] * imgScale
            newimg = cv.resize(crop_img[i], (int(newX), int(newY)))
            cv.imshow("Show by CV2", newimg)\





if __name__ == "__main__":
    d = faceDetect("first")
    d.detectFaceOfImage(d.cascade_path, d.path)



# cv.imshow('img',crop_img)
# cv.waitKey(0)
# cv.destroyAllWindows()















