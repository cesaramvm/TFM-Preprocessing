import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import imutils
from pathlib import Path
from Utils import *


# path = "D:\Máster MUIIA\Prácticas\TFM\siim-isic-melanoma-classification\jpeg"
# path = "G:\Mi unidad\Prácticas\TFM\siim-isic-melanoma-classification\jpeg"
path = "D:\siim-isic-melanoma-classification\jpeg"
path = os.path.join(path)

numProcessed = 0

for root, dirs, files in os.walk(path, topdown=False):
    for fileNameFull in files:
        allProcessStartTime = time.time()
        numProcessed=numProcessed+1
        fileNameOnly = str(Path(fileNameFull).with_suffix(''))
        if fileNameFull == "desktop.ini":
            continue
        filePath = os.path.join(root, fileNameFull)
        savePath = filePath.replace(path, "")
        savePath224 = "processedDataset/jpeg224" + savePath.replace("\\", "/")
        savePath331 = "processedDataset/jpeg331" + savePath.replace("\\", "/")
        if(os.path.isfile(savePath224) and os.path.isfile(savePath331)):
            print(fileNameOnly, " ya se ha preprocesado (",numProcessed,")" )
            continue

        print(fileNameOnly, " procesando (", numProcessed, ")")
        originalImage = cv2.imdecode(np.asarray(bytearray(open(filePath, "rb").read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        resizedOnlyWidthImage = resizeWidthByHeight(originalImage, IMAGE_SIZE_331)
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
            if SHOW_CENTER_CALCULATION:
                cv2.imshow(fileNameFull, removedBordersImage)
                cv2.waitKey(0)

        final_image331 = crop_square(removedHairImageBackup, contoursCenterX, contoursCenterY, cropPixelsW, cropPixelsH, IMAGE_SIZE_331)
        final_image224 = crop_square(removedHairImageBackup, contoursCenterX, contoursCenterY, cropPixelsW, cropPixelsH, IMAGE_SIZE_224)

        screenshotsBasePath = "preproScreens/"
        screenshotsPath = screenshotsBasePath + fileNameOnly
        try:
            os.makedirs(screenshotsPath)
        except FileExistsError:
            pass
        # cv2.imwrite(screenshotsPath+'/1original.jpg', originalImage)
        import time

        saveImgsStartTime = time.time()
        import threading
        p = threading.Thread(target=saveImg, args=(resizedOnlyWidthImage.copy(), screenshotsPath+'/2resizedOnlyWidthImage.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(removedHairImageBackup.copy(), screenshotsPath+'/3removedHairImage.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(removedBordersImageBackup.copy(), screenshotsPath+'/4removedBordersImage.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(grayImage.copy(), screenshotsPath+'/5grayImage.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(blur1.copy(), screenshotsPath+'/6blur1.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(threshold1.copy(), screenshotsPath+'/7threshold1.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(blur2.copy(), screenshotsPath+'/8blur2.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(threshold2.copy(), screenshotsPath+'/9threshold2.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(removedBordersImage.copy(), screenshotsPath+'/10removedBordersImage.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(removedHairImage.copy(), screenshotsPath+'/11removedHairImage.jpg'))
        p.start()
        p = threading.Thread(target=saveImg, args=(final_image331.copy(), screenshotsPath+'/12final.jpg'))
        p.start()
        p = threading.Thread(target=saveComparisonChart, args=(originalImage.copy(), final_image331.copy(), fileNameOnly, screenshotsBasePath + fileNameOnly + "_comparison.jpg"))
        p.start()
        try:
            os.makedirs(os.path.dirname(savePath224))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.dirname(savePath331))
        except FileExistsError:
            pass

        p = threading.Thread(target=saveImg, args=(final_image331.copy(), savePath331))
        p.start()
        p = threading.Thread(target=saveImg, args=(final_image224.copy(), savePath224))
        p.start()
        finishTime = time.time()
        print("--- %s All seconds ---" % (finishTime - allProcessStartTime))
        print("--- %s Save threads seconds ---" % (finishTime - saveImgsStartTime))
        if SHOW_RESULT:
            cv2.imshow(fileNameFull, final_image331)
            cv2.waitKey(0)
            cv2.destroyAllWindows()