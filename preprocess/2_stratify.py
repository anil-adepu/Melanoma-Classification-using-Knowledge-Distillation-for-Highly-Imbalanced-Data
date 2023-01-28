import numpy as np, pandas as pd, os
from keras.preprocessing import image
from PIL import Image
import os, random, cv2, copy
from os import path
import ying, dhe, he, dual

def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    # print(str(len(file_list)))
    return file_list

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

## DATASET: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data
data_type = '.jpg'
csv_path = './Datasets/ISIC2020/'
FOLDS = 15

train= pd.read_csv(csv_path + 'train.csv')
cols= train.columns

class_wise=[]
for i in range(1,3):
    class_wise.append(list(train.loc[train[cols[i]]==1.0].image_name))

for x in class_wise:
    random.shuffle(x)

strat=[[] for i in range(FOLDS)]

for c, x in enumerate(class_wise):
    i=0
    for j in range(FOLDS):
        step = int(np.floor(len(x)/FOLDS))
        k= min(len(x), i + step)
        strat[j]+= x[i:k]
        i=k

    if c:
        for id, y in enumerate(x[i:len(x)]):
            strat[id].append(y)
    else:
        for id, y in enumerate(x[i:len(x)]):
            strat[len(strat)-1-id].append(y)

    print('----------------------------------------------------------------------------', len(x))

cnt=set()
for x in strat:
    for y in x:
        cnt.add(y)
print(len(cnt))

cnt=0
for x in strat:
        cnt += len(x)
print(cnt)
for x in strat:
    print(len(x), len(x)*100/32701)


train_path='./DATASETS/ISIC2020/centre_square_cropped/train/'
files=return_list(train_path, '.jpg')

data_save_path = mk_dir('./DATASETS/ISIC2020/stratified_jpg_256/')

for fold, fold_files in enumerate(strat):
    fold_wise_data_save_path = mk_dir('../DATASETS/ISIC2020/stratified_jpg_256/train' + fold + '/')
    for file in fold_files:
        temp_txt = file + '.jpg'
        org_img = np.asarray(image.load_img(train_path + temp_txt))
        org_img = Image.fromarray(org_img)
        org_img.save(path.join((fold_wise_data_save_path) + file + '.jpg'))
