import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time

IMAGE_SIZE = 768

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])
    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized

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
    
base_path = os.path.join("D:\Máster MUIIA\Prácticas\TFM\siim-isic-melanoma-classification\jpeg")
    
for root, dirs, files in os.walk(base_path, topdown=False):
    for file in files:
        if file == "desktop.ini":
            continue

        filePath = os.path.join(root, file)
        newRoot = root.replace("D:\Máster MUIIA\Prácticas\TFM\siim-isic-melanoma-classification\\", "")
        newFilePath =os.path.join(newRoot, file)
        print(filePath)
        print(newFilePath)
        if os.path.isfile(newFilePath):
            print("La imagen ", file, "ya ha sido procesada")
            continue
        try:
            os.makedirs(newRoot)
        except FileExistsError:
            pass


        stream = open(filePath, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        #cv2.imwrite(file+'_original.jpg', image)
        processedImage =  processImage(image)
        cv2.imwrite(newFilePath, processedImage)

        #plt.subplot(1, 2, 1)
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.axis('off')
        #plt.title('Original : '+ file)
        #plt.subplot(1, 2, 2)
        #plt.imshow(cv2.cvtColor(processedImage, cv2.COLOR_BGR2RGB))
        #plt.axis('off')
        #plt.title('Hair Removed + Resize : '+ file)
        #plt.plot()
        #plt.savefig(file + "_comparison.png")
