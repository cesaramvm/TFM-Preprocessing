import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import imutils
from pathlib import Path
from Utils import *




base_path = os.path.join("D:\Máster MUIIA\Prácticas\TFM\siim-isic-melanoma-classification\jpeg")
numProcessed = 0

for root, dirs, files in os.walk(base_path, topdown=False):
    for fileNameFull in files:
        numProcessed=numProcessed+1
        fileNameOnly = str(Path(fileNameFull).with_suffix(''))
        if fileNameFull == "desktop.ini":
            continue
        filePath = os.path.join(root, fileNameFull)
        savePath = filePath.replace("D:\\Máster MUIIA\\Prácticas\\TFM\\siim-isic-melanoma-classification\\","")
        savePath = "processedDataset/"+savePath.replace("\\", "/")
        if(os.path.isfile(savePath)):
            print(fileNameOnly, " ya se ha preprocesado (",numProcessed,")" )
            continue

        print(fileNameOnly, " procesando (", numProcessed, ")")
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

        cntsInfo = getCntsInfo(cnts, imgWidth, imgHeight, blur1, removedBordersImage, fileNameFull)

        (contoursCenterX, contoursCenterY) = getCenterFromContoursData(cntsInfo, removedBordersImage, fileNameFull, imgWidth)


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


        try:
            os.makedirs(os.path.dirname(savePath))
        except FileExistsError:
            pass

        cv2.imwrite(savePath, final_image)

        if SHOW_IMGS:
            cv2.imshow(fileNameFull, final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()