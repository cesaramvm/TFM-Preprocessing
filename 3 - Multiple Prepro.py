import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import imutils

IMAGE_SIZE = 512

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])
    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized

def resizeOnlyWidth(image,window_height=IMAGE_SIZE):
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    window_width = window_height*aspect_ratio
    return cv2.resize(image, (int(window_width), int(window_height)))

def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    return final_image

def processImage(image, image_size=IMAGE_SIZE):
    resized_image = crop_square(image, image_size)    
    hair_removed_image = hair_remove(resized_image)
    return hair_removed_image


def removeBordersByPercentage(image, percentage=0.1):
    width = image.shape[0]
    height = image.shape[1]
    cropPixelsW = math.trunc(width * percentage)
    cropPixelsH = math.trunc(height * percentage)
    return image[cropPixelsW:width-cropPixelsW, cropPixelsH:height-cropPixelsH]


    
base_path = os.path.join("D:\Máster MUIIA\Prácticas\TFM\siim-isic-melanoma-classification\jpeg")
    
for root, dirs, files in os.walk(base_path, topdown=False):
    for file in files:
        if file == "desktop.ini":
            continue

        filePath = os.path.join(root, file)

        stream = open(filePath, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        resizedOnlyWidthImage =  resizeOnlyWidth(image)
        removedHairImage = hair_remove(resizedOnlyWidthImage)
        removedBordersImage = removeBordersByPercentage(removedHairImage)
        grayImage = grayScale = cv2.cvtColor(removedBordersImage, cv2.COLOR_RGB2GRAY)

        ret,thresh1 = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(grayImage,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(grayImage,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(grayImage,127,255,cv2.THRESH_TOZERO_INV)

        th2 = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)

        ret2,th4 = cv2.threshold(grayImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering

        blur1 = cv2.GaussianBlur(grayImage,(5,5),0)
        ret3,th5 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        blur2 = cv2.GaussianBlur(grayImage,(21,21),0)
        ret3,th6 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


        #Limpieza de threshold con otro blur.
        blur3 = cv2.GaussianBlur(th6,(11,11),0)
        ret3,th7 = cv2.threshold(blur3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite('original.png', image)
        cv2.imwrite('test.png', resizedOnlyWidthImage)
        cv2.imwrite('test2.png', removedHairImage)
        cv2.imwrite('test3.png', removedBordersImage)
        cv2.imwrite('test4.png', grayImage)
        cv2.imwrite('test5.png', thresh1)
        cv2.imwrite('test6.png', thresh2)
        cv2.imwrite('test7.png', thresh3)
        cv2.imwrite('test8.png', thresh4)
        cv2.imwrite('test9.png', thresh5)
        cv2.imwrite('test10.png', th2)
        cv2.imwrite('test11.png', th3)
        cv2.imwrite('test12.png', th4)
        cv2.imwrite('test13.png', blur1)
        cv2.imwrite('test14.png', th5)
        cv2.imwrite('test15.png', blur2)
        cv2.imwrite('test16.png', th6)
        cv2.imwrite('test17.png', blur3)
        cv2.imwrite('test18.png', th7)


        cnts = cv2.findContours(th7.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        print("HAY ", len(cnts), " contours")
        #print(cnts)

        sumPerimeter = 0
        sumMeanColors = 0
        contoursCenterX = 0
        contoursCenterY = 0
        # loop over the contours
        for cnt in cnts:
            perimeter = cv2.arcLength(cnt, False)
            sumPerimeter = sumPerimeter + perimeter
        # loop over the contours
            x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
            #cv2.rectangle(blur2,(x,y),(x+w,y+h),(0,255,0),2)
            meanColor = np.array(cv2.mean(blur2[y:y+h,x:x+w])).astype(np.uint8)
            blackAmmount = 255-meanColor[0]
            print(meanColor[0])
            print(blackAmmount)
            print('Average color (BGR): ',meanColor)
            cv2.imshow('cutted contour',blur2[y:y+h,x:x+w])
            cv2.waitKey(0)

        for c in cnts:
            perimeter = cv2.arcLength(c, False)
            importance = perimeter / sumPerimeter
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            contoursCenterX = int(contoursCenterX + importance*cX)
            contoursCenterY = int(contoursCenterY + importance*cY)
            # draw the contour and center of the shape on the image
            #cv2.drawContours(blur2, [c], -1, (0, 255, 0), 2)
            #cv2.circle(blur2, (cX, cY), 7, (255, 255, 255), -1)
            #cv2.circle(blur2, (contoursCenterX, contoursCenterY), 7, (255, 255, 255), -1)
            #cv2.putText(blur2, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # show the image
            #cv2.imshow("CONT", blur2)
            #cv2.waitKey(0)


        cv2.imwrite('test19.png', blur2)

        #cv2.imwrite('test.png', test)
        #cv2.imwrite('original.jpg', image)