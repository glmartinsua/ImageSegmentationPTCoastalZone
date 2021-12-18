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
# model = load_model('models/b4_sset2_totalloss_192x288_unet_res34_50epochs.hdf5', compile=False)



# define model

# custom loss with weights
CLASS_0_WEIGHT = 0.1
CLASS_1_WEIGHT = 0.2
CLASS_2_WEIGHT = 0.7

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

model.load_weights('models/b4_sset2_totalloss_192x288_unet_res34_50epochs.hdf5')


''' # =================================================================================================================
# Model Testing
================================================================================================================= # '''


# Predict And Evaluate -!

''' Example: GALAPOS 18 (used in training)'''

# TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos18.JPG'
# TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos18.png'

# results (in metrics order): X
# Mean IoU = X
# IoU for class1 (0) is:  X
# IoU for class2 (1) is:  X
# IoU for class3 (2) is:  X


''' Galapos6 '''
# results (in metrics order): [0.8827672004699707, 0.4567564129829407, 0.5480862259864807, 0.3333333432674408]
# Mean IoU = 0.50559914
# IoU for class1 (0) is:  0.20983376241399382
# IoU for class2 (1) is:  0.8462068748546058
# IoU for class3 (2) is:  0.4607566318327974


''' Galapos43 '''
# results (in metrics order): [0.9552672505378723, 0.4225688576698303, 0.5333940386772156, 0.3333333432674408]
# Mean IoU = 0.42173716
# IoU for class1 (0) is:  0.5883010302525464
# IoU for class2 (1) is:  0.6214723673647984
# IoU for class3 (2) is:  0.05543816544677951


''' Other: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos6.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos6.png'

pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image(TEST_IMG_DIR, TEST_MASK_DIR, model, preprocess_input, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
biou.save_predicton_as_image(pred_img, 'test.png')
