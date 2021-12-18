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
#model = load_model('models/b6_sset3_192x288_totalloss_unet_res34_50epochs.hdf5', compile=False)


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

model.load_weights('models/b6_sset3_192x288_totalloss_unet_res34_50epochs.hdf5')


''' ### '''


# Predict And Evaluate -!

''' Exemplo: GALAPOS 18 (usada em treino)'''

# TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos18.JPG'
# TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos18.png'

# results (in metrics order): [0.7492436170578003, 0.8020192384719849, 0.8714717030525208, 0.39775529503822327]
# Mean IoU = 0.8247075
# IoU for class1 (0) is:  0.8568371073254152
# IoU for class2 (1) is:  0.9871249756400284
# IoU for class3 (2) is:  0.6301602997661379


''' Galapos36 '''
# results (in metrics order): [0.7418432831764221, 0.8182240724563599, 0.8916745185852051, 0.3360513150691986]
# Mean IoU = 0.8331723
# IoU for class1 (0) is:  0.9495857090167155
# IoU for class2 (1) is:  0.8818703266323454
# IoU for class3 (2) is:  0.6680608314187441


''' Galapos6 '''
# results (in metrics order): [0.8170881867408752, 0.6722668409347534, 0.7581257224082947, 0.48256054520606995]
# Mean IoU = 0.7428412
# IoU for class1 (0) is:  0.6488532106475533
# IoU for class2 (1) is:  0.9707946497912299
# IoU for class3 (2) is:  0.6088756213773139


''' Galapos43 '''
# results (in metrics order): [0.7418432831764221, 0.8182240724563599, 0.8916745185852051, 0.3360513150691986]
# Mean IoU = X
# IoU for class1 (0) is:  X
# IoU for class2 (1) is:  X
# IoU for class3 (2) is:  X


''' Outros: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos6.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos6.png'



pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image(TEST_IMG_DIR, TEST_MASK_DIR, model, preprocess_input, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
# biou.save_predicton_as_image(pred_img, 'test.png')
