"""
Variaveis do ambiente:
    SM_FRAMEWORK=tf.keras
"""

import segmentation_models as sm
import random
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import load_model
from keras.metrics import MeanIoU
from PIL import Image

# import bioworkutils_WM as biou
from BiologiaWork.WM_WaterMuss import bioworkutils_WM as biou


''' # =================================================================================================================
# Loading dataset 
================================================================================================================= # '''


# Dataset Directory

IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset4_patches_204x307_GalaposX4\images\\'
MASKS_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset4_patches_204x307_GalaposX4\masks\\'

# Image Transformation Parameters + Number of Classes

SIZE_Y = 192  # height
SIZE_X = 288  # width
n_classes = 3

# Data Loading + Verification

X_train, X_val, X_test, y_train, y_train_cat, y_val, y_val_cat, y_test, y_test_cat = biou.load_data(IMG_DIR, MASKS_DIR, n_classes, 0.10, 0.10, True, SIZE_Y, SIZE_X)
print("Class values: ", np.unique(y_val))  # 0 is the background/few unlabeled
print("y_train_cat: ", y_train_cat.shape)
print("y_val_cat: ", y_val_cat.shape)
print("y_test_cat: ", y_test_cat.shape)


''' # =================================================================================================================
# Model Configuration 
================================================================================================================= # '''


# Model Parameters

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), keras.metrics.MeanIoU(num_classes=n_classes)]
activation = 'softmax'
BACKBONE = 'resnet101'

# Preprocess Input

preprocess_input = sm.get_preprocessing(BACKBONE)
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
# X_test = preprocess_input(X_test)

# Model Loading (if required)

# Compile=False Only for Predictions
model = load_model('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5', compile=False)


# visualize images from test set
imagelist = [23, 35, 69, 74]

for num in imagelist:
    test_img_number = num

    test_img = X_test[test_img_number]
    # im1 = Image.fromarray(test_img)
    # im1.save(f'testset_image{num}.png')

    test_img = preprocess_input(test_img)

    ground_truth = y_test[test_img_number]
    test_img_input = np.expand_dims(test_img, 0)

    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # biou.save_predicton_as_image(ground_truth[:, :, 0], f'testset_masks{num}.png')
    biou.save_predicton_as_image(predicted_img, f'b14_prediction{num}.png')
