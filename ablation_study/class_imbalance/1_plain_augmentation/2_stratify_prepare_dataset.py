import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt, cv2
from PIL import Image
from os import path
from copy import deepcopy
import Augmentor
from tensorflow.keras.preprocessing import image

def return_file_list(data_path, data_type):
	file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
	# print(str(len(file_list)))
	return file_list

def mk_dir(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	return dir_path

def rm_rf_dir_inner(dir_path):
	if not dir_path.endswith("/"):
		dir_path += "/"
	os.system(f"rm -rf {dir_path}*")

def removeHair(image):
	grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(17,17))
	blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

	# apply thresholding to blackhat
	_,threshold = cv2.threshold(blackhat,36,255,cv2.THRESH_BINARY)

	final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
	return final_image

def image_augmentor(image_data):
	rm_rf_dir_inner(temp_path)
	image_data = Image.fromarray(image_data)
	image_data.save(path.join(temp_path, ('temp_image' + IMAGE_EXTENSION)))
	p = Augmentor.Pipeline(temp_path)
	p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
	p.flip_left_right(probability=1)
	p.zoom_random(probability=0.7, percentage_area=0.9)
	p.flip_top_bottom(probability=1)
	p.flip_random(1)
	p.skew_tilt(probability=1, magnitude=1)
	p.shear(probability=1, max_shear_left=10, max_shear_right=10)
	p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=10)
	p.sample(0)
	##### copy augmented image from output folder and return
	return np.asarray(image.load_img(f"{temp_path}output/" + str(return_file_list(f"{temp_path}output/", IMAGE_EXTENSION)[0])))

####CLEANUP#####
temp_path = mk_dir("./temp/")
#########################

FOLDS = 15
TARGET_IMG_SIZE = 256
IMAGE_EXTENSION = '.jpg'
IN_PATH_TRAIN = './isic2020_datasets/stratified_jpg_1024_inpainted36/train'
IN_PATH_TEST = './isic2020_datasets/stratified_jpg_1024_inpainted36/test/'

MAIN_OUT = mk_dir(f"./stratified_CB_jpg_{TARGET_IMG_SIZE}_inpainted36")
rm_rf_dir_inner(MAIN_OUT)
MAIN_OUT_TEST = mk_dir(f"./stratified_CB_jpg_{TARGET_IMG_SIZE}_inpainted36/test/")
MAIN_OUT_TRAIN_PATH = f"./stratified_CB_jpg_{TARGET_IMG_SIZE}_inpainted36/train"
MAIN_OUT_TRAIN = [''] * FOLDS


df_isic = pd.read_csv('./isic2020_datasets/stratified_jpg_1024_inpainted36/train_final2.csv')
df_isic_out = deepcopy(df_isic)
print('###################################')
# ['image_name', 'patient_id', 'lesion_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant', 'target', 'tfrecord', 'width', 'height', 'patient_code']
print(list(df_isic.columns))
print('###################################')


# 165*56 + 416*55 = 32120 - class1 melanoma
count_to_165 = 0

###### READING IMAGE DATA FROM FOLDS (15)
for fold in range(0, FOLDS):
	# PREPARE OUTPUT PIPELINE
	MAIN_OUT_TRAIN[fold] = mk_dir(f"{MAIN_OUT_TRAIN_PATH}{fold}")

	# PREPARE INPUT PIPELINE
	files_fold = return_file_list(f"{IN_PATH_TRAIN}{fold}", IMAGE_EXTENSION)
	files_fold = map(lambda img: img.split(IMAGE_EXTENSION)[0], files_fold)

	# PROCESS EACH IMAGE
	for img_name in files_fold:
		df_row = df_isic.loc[df_isic['image_name'] == img_name]
		LABEL = df_row.iloc[0]['target']
		TFREC = df_row.iloc[0]['tfrecord']
		print(f"Class:{LABEL}\tProcessing image: {img_name}...\tFold: {fold} tfrec: {TFREC}")

		img_data = np.asarray(image.load_img(IN_PATH_TRAIN + str(fold) + '/' + img_name + IMAGE_EXTENSION))

		if LABEL == 1:
			count_to_165 += 1
			########################## AUGMENT MELANOMA
			augment_count = 55 if count_to_165 > 165 else 56

			# GENERATE {augment_count} IMAGES FOR EACH MELANOMA IMAGE
			for idx in range(0, augment_count):
				augmented_img_data = deepcopy(img_data)
				augmented_img_name = img_name
				if idx > 0:
					augmented_img_name += ('_' + str(idx + 1))
					df_row_augmented = deepcopy(df_row)
					df_row_augmented['image_name'] += ('_' + str(idx + 1))
					df_isic_out = pd.concat([df_isic_out, df_row_augmented], ignore_index = True)
					augmented_img_data = image_augmentor(augmented_img_data)
					augmented_img_data = np.asarray(deepcopy(augmented_img_data))
					augmented_img_data = cv2.resize(augmented_img_data, (TARGET_IMG_SIZE, TARGET_IMG_SIZE), interpolation = cv2.INTER_AREA)

				augmented_img_data = Image.fromarray(augmented_img_data)
				augmented_img_data.save(path.join(MAIN_OUT_TRAIN[fold] + '/', (augmented_img_name + IMAGE_EXTENSION)))
		else:
			######## DO NOTHING #### SAVE NON-MELANOMA AS IS
			img_data = cv2.resize(img_data, (TARGET_IMG_SIZE, TARGET_IMG_SIZE), interpolation = cv2.INTER_AREA)
			img_data = Image.fromarray(img_data)
			img_data.save(path.join(MAIN_OUT_TRAIN[fold] + '/', img_name + IMAGE_EXTENSION))

df_isic_out.to_csv(f"./stratified_CB_jpg_{TARGET_IMG_SIZE}_inpainted36/train_ablation_class_balanced.csv", index=False)



######################### COPY REMAINING DATA AS IS #########################
test = pd.read_csv('./isic2020_datasets/stratified_jpg_1024_inpainted36/test.csv')
test.to_csv(f"./stratified_CB_jpg_{TARGET_IMG_SIZE}_inpainted36/test.csv", index=False)
sample_submission = pd.read_csv('./isic2020_datasets/stratified_jpg_1024_inpainted36/sample_submission.csv')
sample_submission.to_csv(f"./stratified_CB_jpg_{TARGET_IMG_SIZE}_inpainted36/sample_submission.csv", index=False)

test_imgs = return_file_list(IN_PATH_TEST, IMAGE_EXTENSION)
test_imgs = map(lambda img: img.split(IMAGE_EXTENSION)[0], test_imgs)
for test_img_name in test_imgs:
	print(f"Processing test image: {test_img_name}...")
	test_img_data = np.asarray(image.load_img(IN_PATH_TEST + test_img_name + IMAGE_EXTENSION))
	# test_img_data = np.asarray(removeHair(test_img_data))
	test_img_data = cv2.resize(test_img_data, (TARGET_IMG_SIZE, TARGET_IMG_SIZE), interpolation = cv2.INTER_AREA)
	test_img_data = Image.fromarray(test_img_data)
	test_img_data.save(path.join(MAIN_OUT_TEST, test_img_name + IMAGE_EXTENSION))


