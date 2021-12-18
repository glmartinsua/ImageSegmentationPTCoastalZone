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
IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset3_patches_204x307_GalaposX3\images\\'
MASKS_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset3_patches_204x307_GalaposX3\masks\\'

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
BACKBONE = 'inceptionv3'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)


# define model

# Compile=False Only for Predictions
model = load_model('models/b7_sset3_192x288_totalloss_unet_inceptionv3_50epochs.hdf5', compile=False)

"""
# custom loss with weights
CLASS_0_WEIGHT = 0.1
CLASS_1_WEIGHT = 0.2
CLASS_2_WEIGHT = 0.7

# Segmentation models losses 
dice_loss = sm.losses.DiceLoss(class_weights=np.array([CLASS_0_WEIGHT, CLASS_1_WEIGHT, CLASS_2_WEIGHT]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# MODEL
# metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), keras.metrics.MeanIoU(num_classes=4)]

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation)
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

# print(model.summary())
print("Model Compiled.")

model.load_weights('models/b7_sset3_192x288_totalloss_unet_inceptionv3_50epochs.hdf5')


results = model.evaluate(X_test, y_test_cat)
print(results)
print('loss: ', results[0])
print('iou_score: ', results[1])
print('f1-score: ', results[2])
print('mean_iou: ', results[3])

# loss:  0.7301803827285767
# iou_score:  0.8554994463920593
# f1-score:  0.9136675596237183
# mean_iou:  0.3597071170806885 (???)


# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Mean IoU = 0.85856074 


# To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
print('values shape')
print(values.shape)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])  # background
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])  # water
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])  # mussels

print("IoU for class1 (background) is: ", class1_IoU)
print("IoU for class2 (water) is: ", class2_IoU)
print("IoU for class3 (mussels) is: ", class3_IoU)

# IoU for class1 (background) is:  0.9476860815933414
# IoU for class2 (water) is:  0.9945954058289368
# IoU for class3 (mussels) is:  0.6334007772390188  
"""

#######################################################################

# test random image from validation set

test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]

print(test_img.shape)
test_img_input = np.expand_dims(test_img, 0)
print(test_img_input.shape)

prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

#####################################################################
