"""
Variaveis do ambiente:
    SM_FRAMEWORK=tf.keras
"""

import segmentation_models as sm
from keras.models import load_model
from keras.metrics import MeanIoU
import keras
import numpy as np

# import bioworkutils_WM as biou
from BiologiaWork.WM_WaterMuss import bioworkutils_WM as biou


''' # =================================================================================================================
# Model Configuration
================================================================================================================= # '''

# Image Transformation Parameters + Number of Classes

SIZE_Y = 192  # height
SIZE_X = 288  # width
n_classes = 3

# Model Parameters

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), keras.metrics.MeanIoU(num_classes=n_classes)]
activation = 'softmax'

# Custom Loss Weights

CLASS_0_WEIGHT = 0.2  # Background
CLASS_1_WEIGHT = 0.2  # Water
CLASS_2_WEIGHT = 0.6  # Mussels

# Loss Function

dice_loss = sm.losses.DiceLoss(class_weights=np.array([CLASS_0_WEIGHT, CLASS_1_WEIGHT, CLASS_2_WEIGHT]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Ensemble Optimal Weights

weights = [0.3, 0.5, 0.4]
# weights = [0.3, 0.4, 0.3]


''' # MODEL 1 # MODEL 1 # MODEL 1 # MODEL 1 # MODEL 1 # '''

# Preprocess Input

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# Model Loading (if required)

# Compile=False Only for Predictions
# model1 = load_model('models/b12_sset4_192x288_unet_res34_50epochs.hdf5', compile=False)

# Model Definition

model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model1.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model1.load_weights('models/b12_sset4_192x288_unet_res34_50epochs.hdf5')


''' # MODEL 2 # MODEL 2 # MODEL 2 # MODEL 2 # MODEL 2 # '''

# Preprocess Input

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# Model Loading (if required)

# Compile=False Only for Predictions
# model2 = load_model('models/b13_sset4_192x288_unet_incepv3_50epochs.hdf5', compile=False)

# Model Definition

model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model2.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model2.load_weights('models/b13_sset4_192x288_unet_incepv3_50epochs.hdf5')


''' # MODEL 3 # MODEL 3 # MODEL 3 # MODEL 3 # MODEL 3 # '''

# Preprocess Input

BACKBONE3 = 'resnet101'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# Model Loading (if required)

# Compile=False Only for Predictions
# model3 = load_model('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5', compile=False)

# Model Definition

model3 = sm.PSPNet(BACKBONE3, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model3.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model3.load_weights('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5')


# Building Model and Preprocess Lists

preprocessing_inputs = [preprocess_input1, preprocess_input2, preprocess_input3]
models = [model1, model2, model3]


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


''' Galapos36 (used in training) '''
# results (in metrics order): X
# Mean IoU = X
# IoU for class1 (0) is:  X
# IoU for class2 (1) is:  X
# IoU for class3 (2) is:  X


''' Galapos6 (# Weights: 0.3, 0.4, 0.3) '''
# results (in metrics order): [0.82359794 0.63826441 0.70681474 0.56608204]
# Mean IoU = 0.79129094
# IoU for class1 (0) is:  0.6792740517806309
# IoU for class2 (1) is:  0.9754902098114424
# IoU for class3 (2) is:  0.7191086081085173


''' Galapos43 (# Weights: 0.3, 0.4, 0.3) '''
# results (in metrics order): [0.81044436 0.7141385  0.79003068 0.35505638]
# Mean IoU = 0.81859106
# IoU for class1 (0) is:  0.970373297746117
# IoU for class2 (1) is:  0.9581787635814318
# IoU for class3 (2) is:  0.5272210340278934


''' Galapos6 (# Weights: 0.3, 0.5, 0.4) '''
# results (in metrics order): [0.82479841 0.63531388 0.70301047 0.57263892]
# Mean IoU = 0.79051584
# IoU for class1 (0) is:  0.6764866135153014
# IoU for class2 (1) is:  0.9753751940528405
# IoU for class3 (2) is:  0.7196857235960752


''' Galapos43 (# Weights: 0.3, 0.5, 0.4) '''
# results (in metrics order): [0.81495183 0.70742569 0.78530464 0.35538818]
# Mean IoU = 0.8185838
# IoU for class1 (0) is:  0.970145717356102
# IoU for class2 (1) is:  0.9577573222064656
# IoU for class3 (2) is:  0.5278485651703301


''' Other: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos6.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos6.png'

pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image_ensemble(TEST_IMG_DIR, TEST_MASK_DIR, models, preprocessing_inputs, weights, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
# biou.save_predicton_as_image(pred_img, 'test.png')
