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
BACKBONE = 'resnet101'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Model Loading (if required)

# Compile=False Only for Predictions
# model = load_model('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5', compile=False)


# Custom Loss Weights

CLASS_0_WEIGHT = 0.2  # Background
CLASS_1_WEIGHT = 0.2  # Water
CLASS_2_WEIGHT = 0.6  # Mussels

# Loss Function

dice_loss = sm.losses.DiceLoss(class_weights=np.array([CLASS_0_WEIGHT, CLASS_1_WEIGHT, CLASS_2_WEIGHT]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Model Definition

model = sm.PSPNet(BACKBONE, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model.load_weights('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5')


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
# results (in metrics order): [0.86515212059021, 0.6140658855438232, 0.7485741972923279, 0.3805010914802551]
# Mean IoU = 0.6156702
# IoU for class1 (0) is:  0.6880126447974412
# IoU for class2 (1) is:  0.4715365760067747
# IoU for class3 (2) is:  0.6874613335353459


''' Galapos6 '''
# results (in metrics order): [0.8500757217407227, 0.5323172807693481, 0.5970329642295837, 0.7896276116371155]
# Mean IoU = 0.65375036
# IoU for class1 (0) is:  0.33112704988126934
# IoU for class2 (1) is:  0.9497542246668728
# IoU for class3 (2) is:  0.6803695537177963


''' Galapos43 '''
# results (in metrics order): [0.9334203600883484, 0.5209959149360657, 0.645129382610321, 0.39086639881134033]
# Mean IoU = 0.5232642
# IoU for class1 (0) is:  0.7114819892955151
# IoU for class2 (1) is:  0.6376490222205097
# IoU for class3 (2) is:  0.22066156212497676


''' Other: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos43.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos43.png'

pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image(TEST_IMG_DIR, TEST_MASK_DIR, model, preprocess_input, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
biou.save_predicton_as_image(pred_img, 'test.png')
