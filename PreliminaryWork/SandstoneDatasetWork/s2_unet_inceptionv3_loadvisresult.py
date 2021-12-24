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
BACKBONE = 'inceptionv3'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

# define model

# Compile=False Only for Predictions
model = load_model('models\\unet_inceptionv3_50epochs.hdf5', compile=False)


# visualize images from test set

imagelist = [5, 23, 32, 59, 77, 84]

for num in imagelist:
    test_img_number = num
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_input = np.expand_dims(test_img, 0)

    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # ground_truth = cv2.applyColorMap(ground_truth, cv2.COLORMAP_JET)
    # predicted_img = cv2.applyColorMap(predicted_img, cv2.COLORMAP_JET)

    # cv2.imwrite(f'testset_image{num}.png', test_img[:, :, 0])
    # cv2.imwrite(f'testset_masks{num}.png', ground_truth[:, :, 0])
    # cv2.imwrite(f's1_prediction{num}.png', predicted_img)

    '''
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.axis('off')
    plt.savefig(f'testset_image{num}.png')

    plt.imshow(ground_truth[:, :, 0], cmap='jet')
    plt.axis('off')
    plt.savefig(f'testset_masks{num}.png')
    '''

    plt.imshow(predicted_img, cmap='jet')
    plt.axis('off')
    plt.savefig(f's2_prediction{num}.png')

