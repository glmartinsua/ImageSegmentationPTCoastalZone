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
#model = load_model('models/b10_sset3_192x288_totallossV3_unet_res34_50epochs.hdf5', compile=False)


# define model

# custom loss with weights
CLASS_0_WEIGHT = 0.2
CLASS_1_WEIGHT = 0.2
CLASS_2_WEIGHT = 0.6

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

model.load_weights('models/b10_sset3_192x288_totallossV3_unet_res34_50epochs.hdf5')


''' ### '''


# Predict And Evaluate -!

''' Exemplo: GALAPOS 18 (usada em treino)'''

# TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos18.JPG'
# TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos18.png'

# results (in metrics order): X
# Mean IoU = X
# IoU for class1 (0) is:  X
# IoU for class2 (1) is:  X
# IoU for class3 (2) is:  X


''' Galapos36 '''
# results (in metrics order): [0.7224568724632263, 0.86683189868927, 0.9161841869354248, 0.33341971039772034]
# Mean IoU = 0.88555056
# IoU for class1 (0) is:  0.9795704810767035
# IoU for class2 (1) is:  0.9815517685382947
# IoU for class3 (2) is:  0.6955295011728023


''' Galapos6 '''
# results (in metrics order): [0.7824151515960693, 0.6888685822486877, 0.7639158368110657, 0.35454997420310974]
# Mean IoU = 0.75223595
# IoU for class1 (0) is:  0.6767309672168438
# IoU for class2 (1) is:  0.9667877042847163
# IoU for class3 (2) is:  0.6131891631973883


''' Galapos43 '''
# results (in metrics order): [0.7962068319320679, 0.7233049869537354, 0.7906157374382019, 0.3333333432674408]
# Mean IoU = 0.72912115
# IoU for class1 (0) is:  0.943278808698311
# IoU for class2 (1) is:  0.9258979956708362
# IoU for class3 (2) is:  0.31818660616618727


''' Outros: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos43.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos43.png'


pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image(TEST_IMG_DIR, TEST_MASK_DIR, model, preprocess_input, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
# biou.save_predicton_as_image(pred_img, 'test.png')
