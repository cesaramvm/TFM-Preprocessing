import numpy as np
import cv2
import math


# DRAW_CENTER_CALCULATION = True
DRAW_CENTER_CALCULATION = False or SHOW_CENTER_CALCULATION
# DRAW_CONTOURS = True
DRAW_CONTOURS = False or SHOW_CONTOURS


def crop_square(img, cx, cy, cropPixelsW, cropPixelsH, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])
    if cx == 0 or cy == 0:
        cx = int(w / 2)
        # cy = int(h / 2)
    else:
        cx = cx + cropPixelsW
        # cy = cy + cropPixelsH
        if cx < min_size / 2:
            cx = int(min_size / 2)
        if cx > w - (min_size / 2):
            cx = w - int(min_size / 2)
    # A cy no le hago ni caso porque ya está redimensionado a 512 y no se recorta nada solo se centra el eje X
    cy = int(h / 2)
    # Centralize and crop
    crop_img = img[int(cy - min_size / 2):int(cy + min_size / 2), int(cx - min_size / 2):int(cx + min_size / 2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized


def resizeWidthByHeight(image, window_height):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    window_width = window_height * aspect_ratio
    return cv2.resize(image, (int(window_width), int(window_height)))


def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return final_image


# def processImage(image, image_size):
# resized_image = crop_square(image, image_size)
# hair_removed_image = hair_remove(resized_image)
# return hair_removed_image

def removeBordersByPercentage(image, percentage=REMOVE_BORDERS_PERCENTAGE):
    height = image.shape[0]
    width = image.shape[1]
    cropPixelsHeight = math.trunc(height * percentage)
    cropPixelsWidth = math.trunc(width * percentage)
    return (image[cropPixelsHeight:height - cropPixelsHeight, cropPixelsWidth:width - cropPixelsWidth], cropPixelsWidth,
            cropPixelsHeight)


def checkCntTouchesCorners(cnt, imgWidth, imgHeight):
    lateralPositioned = False
    x, y, w, h = cv2.boundingRect(cnt)
    # touchesTopLeft = x == 0 and y == 0
    # touchesTopRight = x + w == imgWidth and y == 0
    # touchesBottomLeft = x == 0 and y + h == imgHeight
    # touchesBottomRight = x + w == imgWidth and y + h == imgHeight
    # touchesCorner = (x == 0 and y == 0) or (x + w == imgWidth and y == 0) or (x == 0 and y + h == imgHeight) or (x + w == imgWidth and y + h == imgHeight)
    # Pierdo legibilidad pero gano evaluación perezosa.
    if (x == 0 and y == 0) or (x + w == imgWidth and y == 0) or (x == 0 and y + h == imgHeight) or (
            x + w == imgWidth and y + h == imgHeight):
        lateralContourDoubleCheckPixelWidth = 0.15 * imgWidth
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return True  # Área 0...muy feo.
        cX = int(M["m10"] / M["m00"])
        if cX < lateralContourDoubleCheckPixelWidth or cX > imgWidth - lateralContourDoubleCheckPixelWidth:
            return True
        return False
        # lateralPositioned = True
    return False


def getCntsInfo(cnts, imgWidth, imgHeight, blur1, removedBordersImage, fileNameFull):
    numContours = len(cnts)
    sumPerimeter = 0
    sumArea = 0
    sumBlackAmmount = 0
    contoursData = list()
    for cntIndex in range(numContours):
        cnt = cnts[cntIndex]
        if checkCntTouchesCorners(cnt, imgWidth, imgHeight):
            continue
            # pass
        perimeter = cv2.arcLength(cnt, False)
        mask = np.zeros(blur1.shape, np.uint8)
        cv2.drawContours(mask, cnts, cntIndex, 255, -1)
        mean = cv2.mean(blur1, mask=mask)
        blackAmmount = 255 - mean[0]
        # Todo mirar si aqui meter área mejor?
        if blackAmmount < 70 or perimeter < 100:
            continue
            # pass

        M = cv2.moments(cnt)
        area = int(M["m00"])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contoursData.append([perimeter, area, blackAmmount, cX, cY])
        sumArea = sumArea + area
        sumPerimeter = sumPerimeter + perimeter
        sumBlackAmmount = sumBlackAmmount + blackAmmount
        if DRAW_CONTOURS:
            cv2.drawContours(removedBordersImage, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(removedBordersImage, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(removedBordersImage, "B:" + str(int(blackAmmount)), (cX - 20, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 5)
            cv2.putText(removedBordersImage, "B:" + str(int(blackAmmount)), (cX - 20, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            cv2.putText(removedBordersImage, "P:" + str(int(perimeter)), (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0), 5)
            cv2.putText(removedBordersImage, "P:" + str(int(perimeter)), (cX - 20, cY + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255), 2)
        if SHOW_CONTOURS:
            cv2.imshow(fileNameFull, removedBordersImage)
            cv2.waitKey(0)
    return contoursData, sumPerimeter, sumArea, sumBlackAmmount


def getCenterFromContoursData(cntsInfo, removedBordersImage, fileNameFull, imgWidth, showImgs=SHOW_CENTER_CALCULATION):
    (contoursData, sumPerimeter, sumArea, sumBlackAmmount) = cntsInfo
    contoursCenterX = 0
    contoursCenterY = 0
    imgCenter = int(imgWidth / 2)

    contourScoreSum = 1
    contourScores = list()
    if len(contoursData) == 1:
        (perimeter, area, blackAmmount, contourCenterX, contourCenterY) = contoursData[0]
        if DRAW_CENTER_CALCULATION:
            cv2.circle(removedBordersImage, (contourCenterX, contourCenterY), 7, (0, 0, 255), -1)
        if showImgs:
            cv2.imshow(fileNameFull, removedBordersImage)
            cv2.waitKey(0)
        return contourCenterX, contourCenterY
    for data in contoursData:
        (perimeter, area, blackAmmount, contourCenterX, contourCenterY) = data
        distanceFromMiddle = abs(imgCenter - contourCenterX)
        imgCenteredPercentage = 1 - (distanceFromMiddle / (imgCenter * 1.2))
        imgAreaPercentage = area / sumArea
        contourScore = blackAmmount * imgAreaPercentage * imgCenteredPercentage
        contourScores.append(contourScore)
        contourScoreSum = contourScoreSum + contourScore
    for contourIndex in range(len(contoursData)):
        data = contoursData[contourIndex]
        (perimeter, area, blackAmmount, contourCenterX, contourCenterY) = data
        contourScore = contourScores[contourIndex]
        scorePercentage = contourScore / contourScoreSum
        contoursCenterX = int(contoursCenterX + scorePercentage * contourCenterX)
        contoursCenterY = int(contoursCenterY + scorePercentage * contourCenterY)
        if DRAW_CENTER_CALCULATION:
            cv2.circle(removedBordersImage, (contoursCenterX, contoursCenterY), 7, (0, 0, 255), -1)
        if showImgs:
            cv2.imshow(fileNameFull, removedBordersImage)
            cv2.waitKey(0)
    return contoursCenterX, contoursCenterY
