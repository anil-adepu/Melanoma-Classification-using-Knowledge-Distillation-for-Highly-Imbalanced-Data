import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt, cv2
import tensorflow as tf, re, math
from keras.preprocessing import image
from PIL import Image
from os import path

## INPUT DATASET: https://www.kaggle.com/datasets/aniladepu/isic2020-256x256-jpg
GCS_PATH = './stratified_jpg_256/train'
GCS_PATH2 = './stratified_jpg_256/test/'
IMGS2 = os.listdir(GCS_PATH2)
print(len(IMGS2))

print(GCS_PATH, GCS_PATH2)

def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    # print(str(len(file_list)))
    return file_list

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def removeHair(image):
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(17,17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,36,255,cv2.THRESH_BINARY)

    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    return final_image


check = []
tmp = './stratified_jpg_256_inpainted/train'

for i in range(15):
    data_save_path = tmp + str(i)
    data_save_path = mk_dir(data_save_path)
    files = return_list(GCS_PATH + str(i), '.jpg')

    for lineIdx in range(len(files)):
        temp_txt = files[lineIdx]
        print('Processing Img {idx}: {temp_txt}'.format(idx=lineIdx + 1, temp_txt=temp_txt))
        # load image
        check.append(temp_txt)
        org_img = np.asarray(image.load_img(GCS_PATH + str(i) + '/' + temp_txt))
        inpainted_img = removeHair(org_img)
        finalImg = Image.fromarray(inpainted_img)
        finalImg.save(path.join(data_save_path+'/', temp_txt))


print(len(check))


check = []
tmp = mk_dir('./stratified_jpg_256_inpainted/test/')
data_save_path = mk_dir(tmp)

files = return_list(GCS_PATH2, '.jpg')
for lineIdx in range(len(files)):
    temp_txt = files[lineIdx]
    print('Processing Img {idx}: {temp_txt}'.format(idx=lineIdx + 1, temp_txt=temp_txt))
    # load image
    check.append(temp_txt)
    org_img = np.asarray(image.load_img(GCS_PATH2 + temp_txt))
    inpainted_img = removeHair(org_img)
    finalImg = Image.fromarray(inpainted_img)
    finalImg.save(path.join(data_save_path, temp_txt))

print(len(check))

# These final jpeg images are used to generate tfrecords.
## JPEG DATASET: https://www.kaggle.com/datasets/akra98/isic2020-jpg-256x256-inpainted2
## TFREC DATASET: https://www.kaggle.com/datasets/aniladepu/isic2020-tfrec-256x256-inpainted2
## Code: https://www.kaggle.com/code/aniladepu/create-tfrecords/notebook
