import os
import numpy as np
from keras.preprocessing import image
import cv2
from PIL import Image
from os import path
import shutil, copy

def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    # print(str(len(file_list)))
    return file_list

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

data_type = '.jpg'
TARGET_SIZE = 256

## jpeg DATASET: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data
data_img_path_train = './Datasets/ISIC2020/jpeg/train/'
data_img_path_test = './Datasets/ISIC2020/jpeg/test/'

data_save_path = './Datasets/ISIC2020/centre_square_cropped/'
data_save_path_train = mk_dir(data_save_path + 'train/')
data_save_path_test = mk_dir(data_save_path + 'test/')

file_train_list = return_list(data_img_path_train, data_type)
for lineIdx in range(len(file_train_list)):
    temp_txt = data_img_path_train[lineIdx]
    print('Processing Img {idx}: {temp_txt}'.format(idx=lineIdx + 1, temp_txt=temp_txt))
    org_img = np.asarray(image.load_img(data_img_path_train + temp_txt))
    remove=min(org_img.shape[0], org_img.shape[1])//2
    x,y= org_img.shape[0]//2, org_img.shape[1]//2
    # centred square crop
    org_img = org_img[x-remove:x+remove, y-remove:y+remove]
    # resize
    org_img = cv2.resize(org_img, (TARGET_SIZE, TARGET_SIZE), interpolation = cv2.INTER_AREA)
   
    cropImg = Image.fromarray(org_img)
    cropImg.save(path.join(data_save_path_train, temp_txt[:-4] + '.jpg'))



file_test_list = return_list(data_img_path_test, data_type)
for lineIdx in range(len(file_test_list)):
    temp_txt = data_img_path_test[lineIdx]
    print('Processing Img {idx}: {temp_txt}'.format(idx=lineIdx + 1, temp_txt=temp_txt))
    org_img = np.asarray(image.load_img(data_img_path_test + temp_txt))
    remove=min(org_img.shape[0], org_img.shape[1])//2
    x,y= org_img.shape[0]//2, org_img.shape[1]//2
    # centred square crop
    org_img = org_img[x-remove:x+remove, y-remove:y+remove]
    # resize
    org_img = cv2.resize(org_img, (TARGET_SIZE, TARGET_SIZE), interpolation = cv2.INTER_AREA)
   
    cropImg = Image.fromarray(org_img)
    cropImg.save(path.join(data_img_path_test, temp_txt[:-4] + '.jpg'))
