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
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Model Loading (if required)

# Compile=False Only for Predictions
# model = load_model('models/b12_sset4_192x288_unet_res34_50epochs.hdf5', compile=False)


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

model.load_weights('models/b12_sset4_192x288_unet_res34_50epochs.hdf5')


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
# results (in metrics order): [0.725226104259491, 0.8675883412361145, 0.9114598631858826, 0.3739154636859894]
# Mean IoU = 0.8835797
# IoU for class1 (0) is:  0.9790911699661909
# IoU for class2 (1) is:  0.9743468713444292
# IoU for class3 (2) is:  0.6973009041210281


''' Galapos6 '''
# results (in metrics order): [0.8132159113883972, 0.6501588821411133, 0.7310898900032043, 0.4862509071826935]
# Mean IoU = 0.74470377
# IoU for class1 (0) is:  0.619531219294913
# IoU for class2 (1) is:  0.9691598570445743
# IoU for class3 (2) is:  0.6454204572622149


''' Galapos43 '''
# results (in metrics order): [0.7793169021606445, 0.7571625709533691, 0.81734699010849, 0.3526586890220642]
# Mean IoU = 0.7784279
# IoU for class1 (0) is:  0.9567066419471038
# IoU for class2 (1) is:  0.931266333188124
# IoU for class3 (2) is:  0.447310739473442


''' Other: '''

TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos43.JPG'
TEST_MASK_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos43.png'

pred_img, split_pred, results, values = biou.predict_and_evaluate_large_image(TEST_IMG_DIR, TEST_MASK_DIR, model, preprocess_input, 3, 204, 307, 0, True, SIZE_Y, SIZE_X)
# biou.save_predicton_as_image(pred_img, 'test.png')
