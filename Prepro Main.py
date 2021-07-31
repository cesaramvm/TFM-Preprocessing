import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import imutils

IMAGE_SIZE = 512
REMOVE_BORDERS_PERCENTAGE = 0.1


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


def removeBordersByPercentage(image, percentage=REMOVE_BORDERS_PERCENTAGE):
    width = image.shape[0]
    height = image.shape[1]
    cropPixelsW = math.trunc(width * percentage)
    cropPixelsH = math.trunc(height * percentage)
    return image[cropPixelsW:width-cropPixelsW, cropPixelsH:height-cropPixelsH]


base_path = os.path.join("D:\Máster MUIIA\Prácticas\TFM\siim-isic-melanoma-classification\jpeg")


def checkCntTouchesCorners(cnt, imgWidth, imgHeight):
    x, y, w, h = cv2.boundingRect(cnt)
    touchesTopLeft = x == 0 and y == 0
    touchesTopRight = x + w == imgWidth and y == 0
    touchesBottomLeft = x == 0 and y + h == imgHeight
    touchesBottomRight = x + w == imgWidth and y + h == imgHeight
    return touchesTopLeft or touchesTopRight or touchesBottomLeft or touchesBottomRight


for root, dirs, files in os.walk(base_path, topdown=False):
    for file in files:
        if file == "desktop.ini":
            continue

        filePath = os.path.join(root, file)

        stream = open(filePath, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        resizedOnlyWidthImage = resizeOnlyWidth(image)
        removedHairImage = hair_remove(resizedOnlyWidthImage)
        removedBordersImage = removeBordersByPercentage(removedHairImage)

        grayImage = grayScale = cv2.cvtColor(removedBordersImage, cv2.COLOR_RGB2GRAY)
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
        cv2.imwrite('test15.png', blur2)
        cv2.imwrite('test16.png', th6)
        cv2.imwrite('test17.png', blur3)
        cv2.imwrite('test18.png', th7)


        cnts = cv2.findContours(th7.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        numContours = len(cnts)
        print("HAY ", numContours, " contours")

        imgHeight, imgWidth = th7.shape

        sumPerimeter = 0
        sumBlackAmmount = 0
        contourData = list()
        for cntIndex in range(numContours):
            cnt = cnts[cntIndex]
            if checkCntTouchesCorners(cnt, imgWidth, imgHeight):
                print("Ese contorno toca alguna esquina")

            perimeter = cv2.arcLength(cnt, False)

            mask = np.zeros(blur2.shape, np.uint8)
            cv2.drawContours(mask, cnts, cntIndex, 255, -1)
            mean = cv2.mean(blur2, mask=mask)
            blackAmmount = 255-mean[0]
            if(blackAmmount<70 or perimeter<100): continue


            M = cv2.moments(cnt)
            area = int(M["m00"])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            contourData.append([perimeter, blackAmmount, cX, cY])

            sumPerimeter = sumPerimeter + perimeter
            sumBlackAmmount = sumBlackAmmount + blackAmmount
            
            # cv2.rectangle(removedBordersImage,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.circle(removedBordersImage, (int(xcircle), int(ycircle)), int(radius), (0, 255, 0), 2)
            #cv2.ellipse(removedBordersImage,ellipse,(0,255,0),2)
            #cv2.imshow('cutted contour',removedBordersImage[y:y+h,x:x+w])
            #cv2.waitKey(0)
            #print(cnt)
            # draw the contour and center of the shape on the image
            cv2.drawContours(removedBordersImage, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(removedBordersImage, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(removedBordersImage, "B:" + str(int(blackAmmount)), (cX - 20, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
            cv2.putText(removedBordersImage, "B:" + str(int(blackAmmount)), (cX - 20, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(removedBordersImage, "P:" + str(int(perimeter)), (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), 5)
            cv2.putText(removedBordersImage, "P:" + str(int(perimeter)), (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)

            cv2.imshow(file, removedBordersImage)
            cv2.waitKey(0)

        contoursCenterX = 0
        contoursCenterY = 0
        blackAmmountPercentage = 0.5
        perimeterPercentage = 1-blackAmmountPercentage
        for data in contourData:
            perimeter = data[0]
            blackAmmount = data[1]
            contourCenterX = data[2]
            contourCenterY = data[3] 


            perimeterImportance = perimeter / sumPerimeter
            blackImportance = blackAmmount / sumBlackAmmount
            totaImportance = blackAmmountPercentage*blackImportance + perimeterPercentage*perimeterImportance

            contoursCenterX = int(contoursCenterX + totaImportance*contourCenterX)
            contoursCenterY = int(contoursCenterY + totaImportance*contourCenterY)


            # draw the contour and center of the shape on the image
            #cv2.drawContours(removedBordersImage, [c], -1, (0, 255, 0), 2)
            #cv2.circle(removedBordersImage, (cX, cY), 7, (255, 255, 255), -1)
            cv2.circle(removedBordersImage, (contoursCenterX, contoursCenterY), 7, (0, 0, 255), -1)
            # show the image
            cv2.imshow(file, removedBordersImage)
            cv2.waitKey(0)
        if(contoursCenterX == 0):
            text = "No contours detected" if numContours==0 else str(numContours) + " contours detected but discarted"
            cv2.putText(removedBordersImage, text, (int(imgWidth/2), int(imgHeight/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
            cv2.putText(removedBordersImage, text, (int(imgWidth / 2), int(imgHeight / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)

            cv2.drawContours(removedBordersImage, cnts, -1, 255, -1)
            cv2.imshow(file, removedBordersImage)
            cv2.waitKey(0)

        cv2.destroyAllWindows()