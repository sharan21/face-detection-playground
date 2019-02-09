import numpy as np
import cv2 as cv
import subprocess


class detectFace:

    cascade_path = 'haarcascade_frontalface_default.xml'
    path = 'sample_pictures/group1.jpg'
    imagewidth = 100;
    imageheight = 100;
    crop_img = []
    gray_img = []


    def __init__(self, instance):

        self.instance = instance
        print("Object initialised!")
        print()

    def detectFaceOfImage(self,cascade_path, path): # works on only 1 image per function call, does not work on direcotries

        print ("Detecting facing...")
        print ()

        img = cv.imread(path)

        face_cascade = cv.CascadeClassifier(cascade_path)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.5, 1)

        for (x, y, w, h) in faces:

            # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            self.crop_img.append(img[y:y + h, x:x + w])
            self.gray_img.append(gray[y:y + h, x:x + w])

        print ("Finished detecting faces, stored in crop_img and gray_img...")
        print (0)

        self.crop_img, self.gray_img = self.checkAndFixSize(self.crop_img, self.gray_img)
        temp = []

        # self.exportImages(crop_img, gray_img)
        # self.normalizeImages(self.gray_img)
        self.exportImages(self.crop_img, self.gray_img)

    def checkAndFixSize(self, crop_img, gray_img):
        print ("Scaling the images to proper sizes...")
        print ()
        crop_img_new = []
        gray_img_new = []

        for i in range(len(crop_img)):
            # height, width, depth = crop_img[i].shape
            # imgScale = float(self.imagewidth) / float(width)
            # print("image to be scaled by:", imgScale)
            crop_img_new.append(cv.resize(crop_img[i], (self.imagewidth, self.imageheight)))
            gray_img_new.append(cv.resize(gray_img[i], (self.imagewidth, self.imageheight)))

        print("Done!")

        return crop_img_new, gray_img_new



    def displayCropped(self, crop_img):
        for i in range(len(crop_img)):
            newimg = cv.resize(crop_img[-1], (int(100), int(100)))
            cv.imshow("Show by CV2", newimg)
            cv.waitKey(1000)

        print("Done!")


    def squareTheImages(self, crop_img):
        print("Squaring the images, to conserve aspect ratio while fixing size...")
        print()

        for i in range(len(crop_img)):
            print (crop_img[i].shape)
            height, width, depth = crop_img[i].shape

            if height != width :
                print("{}th image is unsquare!").format(i)
                diff = width - height
                if(diff > 0):
                    print("width excess!")
                    crop_img[i] = crop_img[i][diff/2:width-diff/2,:]
                else:
                    print("width excess!")
                    crop_img[i] = crop_img[i][:,diff / 2:width - diff / 2]



        print("Done!")

        return crop_img

    def assertSquareSize(self, crop_img):
        print ("Asserting that all images are square size...")
        print ()
        for i in range(len(crop_img)):
            height, width, depth = crop_img[i].shape
            if height != width:
                return False

        return True

    def printSize(self,crop_img):

        print ("Printing sizes of all images...")
        print ()

        for i in range(len(crop_img)):
            print("shape of the {}th image is".format(i + 1))
            print (crop_img[i].shape)

        print("Done!")

    def exportImages(self, crop_img, gray_img):

        print ("Writing cropped images into directories...")
        print ()

        for i in range(0, len(crop_img), 1):
            filename = "cropped_images/img{}.jpg".format(i)
            filenamegray = "cropped_gray_images/grayimg{}.jpg".format(i)
            cv.imwrite(filename, crop_img[i])
            cv.imwrite(filenamegray, gray_img[i])

        print("Done!")


    def normalizeImages(self, gray_img_here):
        print ("Normalizing the gray images...")
        print ()
        gray_img_numpy = np.array(gray_img_here)
        for i in range(len(gray_img_here)):
            print
            # print "mean of the {}th image", np.mean(gray_img_numpy[i])
            # print "std dev. of the {}th image", np.std(gray_img_numpy[i])
            # print
            gray_img_here[i] = float(gray_img_here[i] - np.mean(gray_img_numpy[i])) / float(np.std(gray_img_numpy[i], axis=0))

        return gray_img_here


    def deleteAllImages(self):

        print ("Deleting all image files...")
        print ()

        subprocess.Popen(["bash", "./remove-all.sh"])




if __name__ == "__main__":
    d = detectFace("first")
    d.detectFaceOfImage(d.cascade_path, d.path)
    # d.deleteAllImages()
    print(d.gray_img)


"""
self.squareTheImages(crop_img)
if(self.assertSquareSize(crop_img)):s
    print "all square"
else:
    print "something went wrong in square the images function"

self.printSize(crop_img)
"""

















