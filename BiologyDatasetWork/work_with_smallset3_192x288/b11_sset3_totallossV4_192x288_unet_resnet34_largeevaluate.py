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
#model = load_model('models/b11_sset3_192x288_totallossV4_unet_res34_50epochs.hdf5', compile=False)


# define model

# custom loss with weights
CLASS_0_WEIGHT = 0.3
CLASS_1_WEIGHT = 0.2
CLASS_2_WEIGHT = 0.5

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

model.load_weights('models/b11_sset3_192x288_totallossV4_unet_res34_50epochs.hdf5')


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
# results (in metrics order): [0.7463517189025879, 0.8042897582054138, 0.8680687546730042, 0.3387977182865143]
# Mean IoU = 0.8072671
# IoU for class1 (0) is:  0.9518815798386394
# IoU for class2 (1) is:  0.983388215122821
# IoU for class3 (2) is:  0.48653142452234854


''' Galapos6 '''
# results (in metrics order): [0.7824151515960693, 0.6888685822486877, 0.7639158368110657, 0.35454997420310974]
# Mean IoU = 0.7457967
# IoU for class1 (0) is:  0.6633475671098947
# IoU for class2 (1) is:  0.9703551177350067
# IoU for class3 (2) is:  0.6036875327884111


''' Outros: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos6.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos6.png'



pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image(TEST_IMG_DIR, TEST_MASK_DIR, model, preprocess_input, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
biou.save_predicton_as_image(pred_img, 'test.png')
