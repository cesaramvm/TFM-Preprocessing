import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import imutils
from pathlib import Path

IMAGE_SIZE = 512
REMOVE_BORDERS_PERCENTAGE = 0.1


def crop_square(img,cx,cy, cropPixelsW, cropPixelsH, size=IMAGE_SIZE, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])
    if(cx == 0 or cy == 0):
        cx = int(w / 2)
        cy = int(h / 2)
    else:
        cx = cx + cropPixelsW
        cy = cy + cropPixelsH
        if(cx<min_size/2):
            cx = int(min_size / 2)
        if (cx > w-(min_size/2)):
            cx = w-int(min_size/2)
    #A cy no le hago ni caso porque ya está redimensionado a 512 y no se recorta nada solo se centra el eje X
    cy = int(h / 2)
    # Centralize and crop
    crop_img = img[int(cy-min_size/2):int(cy+min_size/2), int(cx-min_size/2):int(cx+min_size/2)]

    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized

def resizeWidthByHeight(image, window_height=IMAGE_SIZE):
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
    height = image.shape[0]
    width = image.shape[1]
    cropPixelsHeight = math.trunc(height * percentage)
    cropPixelsWidth = math.trunc(width * percentage)
    return (image[cropPixelsHeight:width-cropPixelsHeight, cropPixelsWidth:width-cropPixelsWidth],cropPixelsWidth,cropPixelsHeight)





def checkCntTouchesCorners(cnt, imgWidth, imgHeight):
    x, y, w, h = cv2.boundingRect(cnt)
    touchesTopLeft = x == 0 and y == 0
    touchesTopRight = x + w == imgWidth and y == 0
    touchesBottomLeft = x == 0 and y + h == imgHeight
    touchesBottomRight = x + w == imgWidth and y + h == imgHeight
    return touchesTopLeft or touchesTopRight or touchesBottomLeft or touchesBottomRight


def getCntsInfo(cnts):

    numContours = len(cnts)
    sumPerimeter = 0
    sumArea = 0
    sumBlackAmmount = 0
    contourData = list()
    for cntIndex in range(numContours):
        cnt = cnts[cntIndex]
        if checkCntTouchesCorners(cnt, imgWidth, imgHeight):
            continue

        perimeter = cv2.arcLength(cnt, False)

        mask = np.zeros(blur1.shape, np.uint8)
        cv2.drawContours(mask, cnts, cntIndex, 255, -1)
        mean = cv2.mean(blur1, mask=mask)
        blackAmmount = 255 - mean[0]
        if (blackAmmount < 70 or perimeter < 100): continue

        M = cv2.moments(cnt)
        area = int(M["m00"])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        contourData.append([perimeter, area, blackAmmount, cX, cY])

        sumArea = sumArea + area
        sumPerimeter = sumPerimeter + perimeter
        sumBlackAmmount = sumBlackAmmount + blackAmmount

        # draw the contour and center of the shape on the image
        cv2.drawContours(removedBordersImage, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(removedBordersImage, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(removedBordersImage, "B:" + str(int(blackAmmount)), (cX - 20, cY - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 5)
        cv2.putText(removedBordersImage, "B:" + str(int(blackAmmount)), (cX - 20, cY - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        cv2.putText(removedBordersImage, "P:" + str(int(perimeter)), (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 5)
        cv2.putText(removedBordersImage, "P:" + str(int(perimeter)), (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        if SHOW_IMGS:
            cv2.imshow(fileNameFull, removedBordersImage)
            cv2.waitKey(0)
    return (contourData, sumPerimeter, sumArea, sumBlackAmmount)

base_path = os.path.join("D:\Máster MUIIA\Prácticas\TFM\siim-isic-melanoma-classification\jpeg")
SHOW_IMGS = False

for root, dirs, files in os.walk(base_path, topdown=False):
    for fileNameFull in files:
        fileNameOnly = str(Path(fileNameFull).with_suffix(''))
        if fileNameFull == "desktop.ini":
            continue
        filePath = os.path.join(root, fileNameFull)
        originalImage = cv2.imdecode(np.asarray(bytearray(open(filePath, "rb").read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        resizedOnlyWidthImage = resizeWidthByHeight(originalImage)
        removedHairImage = hair_remove(resizedOnlyWidthImage)
        removedHairImageBackup = removedHairImage.copy()
        (removedBordersImage,cropPixelsW,cropPixelsH) = removeBordersByPercentage(removedHairImage)
        removedBordersImageBackup = removedBordersImage.copy()
        imgHeight, imgWidth,_ = removedBordersImage.shape
        grayImage = cv2.cvtColor(removedBordersImage, cv2.COLOR_RGB2GRAY)
        blur1 = cv2.GaussianBlur(grayImage, (21, 21), 0)
        ret3, threshold1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #Limpieza de threshold con otro blur.
        # TODO ver si merece aqui la pena meter un threshold manual agresivo para quitarme cosas.
        blur2 = cv2.GaussianBlur(threshold1, (11, 11), 0)
        ret3, threshold2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cnts = cv2.findContours(threshold2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        numContours = len(cnts)

        (contourData, sumPerimeter, sumArea, sumBlackAmmount) = getCntsInfo(cnts)


        contoursCenterX = 0
        contoursCenterY = 0
        blackAmmountPercentage = 0.2
        perimeterPercentage = 1-blackAmmountPercentage
        for data in contourData:
            #Valorar el tema de que si hay algo centrado se quede centrado?
            #TODO cambiar perímetro a área???
            (perimeter,area,blackAmmount,contourCenterX,contourCenterY) = data
            perimeterImportance = perimeter / sumPerimeter
            blackImportance = blackAmmount / sumBlackAmmount
            totaImportance = blackAmmountPercentage*blackImportance + perimeterPercentage*perimeterImportance

            contoursCenterX = int(contoursCenterX + totaImportance*contourCenterX)
            contoursCenterY = int(contoursCenterY + totaImportance*contourCenterY)
            cv2.circle(removedBordersImage, (contoursCenterX, contoursCenterY), 7, (0, 0, 255), -1)
            # show the image
            if SHOW_IMGS:
                cv2.imshow(fileNameFull, removedBordersImage)
                cv2.waitKey(0)
        if(contoursCenterX == 0):
            text = "No contours detected" if numContours==0 else str(numContours) + " contours detected but discarted"
            cv2.drawContours(removedBordersImage, cnts, -1, 255, -1)
            cv2.putText(removedBordersImage, text, (int(imgWidth/2), int(imgHeight/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5)
            cv2.putText(removedBordersImage, text, (int(imgWidth / 2), int(imgHeight / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
            if SHOW_IMGS:
                cv2.imshow(fileNameFull, removedBordersImage)
                cv2.waitKey(0)

        final_image = crop_square(removedHairImageBackup, contoursCenterX, contoursCenterY, cropPixelsW, cropPixelsH)

        screenshotsBasePath = "preproScreens/"
        screenshotsPath = screenshotsBasePath + fileNameOnly
        try:
            os.makedirs(screenshotsPath)
        except FileExistsError:
            pass
        # cv2.imwrite(screenshotsPath+'/1original.jpg', originalImage)
        cv2.imwrite(screenshotsPath+'/2resizedOnlyWidthImage.jpg', resizedOnlyWidthImage)
        cv2.imwrite(screenshotsPath+'/3removedHairImage.jpg', removedHairImageBackup)
        cv2.imwrite(screenshotsPath+'/4removedBordersImage.jpg', removedBordersImageBackup)
        cv2.imwrite(screenshotsPath+'/5grayImage.jpg', grayImage)
        cv2.imwrite(screenshotsPath+'/6blur1.jpg', blur1)
        cv2.imwrite(screenshotsPath+'/7threshold1.jpg', threshold1)
        cv2.imwrite(screenshotsPath+'/8blur2.jpg', blur2)
        cv2.imwrite(screenshotsPath+'/9threshold2.jpg', threshold2)
        cv2.imwrite(screenshotsPath+'/10removedBordersImage.jpg', removedBordersImage)
        cv2.imwrite(screenshotsPath+'/11removedHairImage.jpg', removedHairImage)
        cv2.imwrite(screenshotsPath+'/12final.jpg', final_image)


        savePath = filePath.replace("D:\\Máster MUIIA\\Prácticas\\TFM\\siim-isic-melanoma-classification\\","")
        savePath = "processedDataset/"+savePath.replace("\\", "/")
        try:
            os.makedirs(os.path.dirname(savePath))
        except FileExistsError:
            pass

        cv2.imwrite(savePath, final_image)

        fig = plt.figure()
        fig.suptitle(fileNameOnly)
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Original:')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Preprocessed:')
        plt.plot()
        plt.savefig(screenshotsBasePath + fileNameOnly + "_comparison.jpg")
        plt.close()
        if SHOW_IMGS:
            cv2.imshow(fileNameFull, final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()