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
from PIL import Image
import keras
from keras.models import load_model
from keras.utils import normalize
from keras.metrics import MeanIoU

# import bioworkutils_WM as biou
from BiologiaWork.WM_WaterMuss import bioworkutils_WM as biou


SIZE_Y = 192  # 204 # height
SIZE_X = 288  # 307 # width
n_classes = 3

activation = 'softmax'
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Compile=False Only for Predictions
model = load_model('models/b9_sset3_192x288_totallossV2_unet_res34_50epochs.hdf5', compile=False)


'''
# define model

# custom loss with weights
CLASS_0_WEIGHT = 0.1
CLASS_1_WEIGHT = 0.1
CLASS_2_WEIGHT = 0.8

# Segmentation models losses 
dice_loss = sm.losses.DiceLoss(class_weights=np.array([CLASS_0_WEIGHT, CLASS_1_WEIGHT, CLASS_2_WEIGHT]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# MODEL
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), keras.metrics.MeanIoU(num_classes=4)]

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation)
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

# print(model.summary())
print("Model Compiled.")

model.load_weights('models/b9_sset3_192x288_totallossV2_unet_res34_50epochs.hdf5')
'''



TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 1\Galapos4.JPG'


pred_img, split_pred = biou.predict_large_image(TEST_IMG_DIR, model, preprocess_input, 204, 307, 0, True, SIZE_Y, SIZE_X)

biou.save_predicton_as_image(pred_img, 'test4.png')
