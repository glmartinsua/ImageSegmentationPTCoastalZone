"""
Variaveis do ambiente:
    SM_FRAMEWORK=tf.keras
"""

import tensorflow as tf
import segmentation_models as sm
import random
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import load_model
from keras.utils import normalize
from keras.metrics import MeanIoU

# import bioworkutils_WM as biou
from BiologiaWork.WM_WaterMuss import bioworkutils_WM as biou


# DATASET DIRECTORIES
IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset1_patches_204x307_FozDoArelho65_Galapos1\images\\'
MASKS_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset1_patches_204x307_FozDoArelho65_Galapos1\masks\\'

SIZE_Y = 192  # 204 # height
SIZE_X = 288  # 307 # width
n_classes = 3

X_train, X_val, X_test, y_train, y_train_cat, y_val, y_val_cat, y_test, y_test_cat = biou.load_data(IMG_DIR, MASKS_DIR, n_classes, 0.10, 0.10, True, SIZE_Y, SIZE_X)
print("Class values in the dataset are ... ", np.unique(y_val))  # 0 is the background/few unlabeled
print("y_train_cat: ", y_train_cat.shape)
print("y_val_cat: ", y_val_cat.shape)
print("y_test_cat: ", y_test_cat.shape)


# MODEL
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), keras.metrics.MeanIoU(num_classes=4)]

activation = 'softmax'
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

# define model

# Compile=False Only for Predictions
model = load_model('models\\b1_sset1_192x288_unet_res34_50epochs.hdf5', compile=False)


'''
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

# print(model.summary())
print("Model Compiled.")

model.load_weights('models\\b1_sset1_192x288_unet_res34_50epochs.hdf5')
'''


# TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos1.JPG'
TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Foz do Arelho\FozDoArelho65.JPG'
TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 1\Galapos6.JPG'

# pred_img, split_pred = biou.predict_large_image(TEST_IMG_DIR, model, 204, 307, 0, True, SIZE_Y, SIZE_X)

pred_img, split_pred = biou.predict_large_image(TEST_IMG_DIR, model, preprocess_input, 204, 307, 0, True, SIZE_Y, SIZE_X)
biou.save_predicton_as_image(pred_img, 'test.png')
