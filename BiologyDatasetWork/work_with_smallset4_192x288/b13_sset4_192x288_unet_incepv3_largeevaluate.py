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

# Model Parameters + Preprocess

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), keras.metrics.MeanIoU(num_classes=n_classes)]
activation = 'softmax'
BACKBONE = 'inceptionv3'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Model Loading (if required)

# Compile=False Only for Predictions
# model = load_model('models/b13_sset4_192x288_unet_incepv3_50epochs.hdf5', compile=False)


# Custom Loss Weights

CLASS_0_WEIGHT = 0.2  # Background
CLASS_1_WEIGHT = 0.2  # Water
CLASS_2_WEIGHT = 0.6  # Mussels

# Loss Function

dice_loss = sm.losses.DiceLoss(class_weights=np.array([CLASS_0_WEIGHT, CLASS_1_WEIGHT, CLASS_2_WEIGHT]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Model Definition

model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model.load_weights('models/b13_sset4_192x288_unet_incepv3_50epochs.hdf5')


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
# results (in metrics order): [0.7026267647743225, 0.9054325222969055, 0.946085512638092, 0.34603944420814514]
# Mean IoU = 0.9101023
# IoU for class1 (0) is:  0.9845364759712231
# IoU for class2 (1) is:  0.9844715740857574
# IoU for class3 (2) is:  0.7612986718303819


''' Galapos6 '''
# results (in metrics order): [0.8115261197090149, 0.7088038921356201, 0.7709447145462036, 0.6547028422355652]
# Mean IoU = 0.7978577
# IoU for class1 (0) is:  0.7509534094517358
# IoU for class2 (1) is:  0.9812293775882992
# IoU for class3 (2) is:  0.6613903275220658


''' Galapos43 '''
# results (in metrics order): [0.7415579557418823, 0.8267273902893066, 0.8782194256782532, 0.34795138239860535]
# Mean IoU = 0.8600016
# IoU for class1 (0) is:  0.9766930856957926
# IoU for class2 (1) is:  0.9616245919792377
# IoU for class3 (2) is:  0.6416871109777916


''' Other: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos6.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos6.png'

pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image(TEST_IMG_DIR, TEST_MASK_DIR, model, preprocess_input, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
# biou.save_predicton_as_image(pred_img, 'test.png')
