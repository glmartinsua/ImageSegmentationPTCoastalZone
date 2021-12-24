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
import sandstoneutils as su

SIZE_X = 128
SIZE_Y = 128
n_classes = 4


X_train, X_val, X_test, y_train, y_train_cat, y_val, y_val_cat, y_test, y_test_cat = su.load_data(4, 0.10, 0.10)
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
model = load_model('models\\unet_res34_50epochs.hdf5', compile=False)


'''
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

# print(model.summary())
print("Model Compiled.")

model.load_weights('models\\unet_res34_50epochs.hdf5')


results = model.evaluate(X_test, y_test_cat)
print(results)
print('loss: ', results[0])
print('iou_score: ', results[1])
print('f1-score: ', results[2])
print('mean_iou: ', results[3])

# loss:  0.060370903462171555
# iou_score:  0.9158669710159302
# f1-score:  0.9547479748725891
# mean_iou:  0.43751049041748047 (???)


# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Mean IoU = 0.91682434 (???) -> Sera apenas o IoU?


# To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

# IoU for class1 is:  0.9087381948776004
# IoU for class2 is:  0.8079062564883382
# IoU for class3 is:  0.9837074746858239
# IoU for class4 is:  0.9669453717280754
'''

#######################################################################

# test random image from validation set

test_img_number = random.randint(0, len(X_val) - 1)
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
