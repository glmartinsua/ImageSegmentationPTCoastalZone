"""
Variaveis do ambiente:
    SM_FRAMEWORK=tf.keras
"""

import segmentation_models as sm
from keras.models import load_model
from keras.metrics import MeanIoU
import keras

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

'''
# Custom Loss Weights

CLASS_0_WEIGHT = 0.2  # Background
CLASS_1_WEIGHT = 0.2  # Water
CLASS_2_WEIGHT = 0.6  # Mussels

# Loss Function

dice_loss = sm.losses.DiceLoss(class_weights=np.array([CLASS_0_WEIGHT, CLASS_1_WEIGHT, CLASS_2_WEIGHT]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
'''

# Ensemble Optimal Weights

weights = [0.3, 0.5, 0.4]
# weights = [0.3, 0.4, 0.3]


''' # MODEL 1 # MODEL 1 # MODEL 1 # MODEL 1 # MODEL 1 # '''

# Preprocess Input

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# Model Loading (if required)

# Compile=False Only for Predictions
model1 = load_model('models/b12_sset4_192x288_unet_res34_50epochs.hdf5', compile=False)
'''
# Model Definition

model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model1.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model1.load_weights('models/b12_sset4_192x288_unet_res34_50epochs.hdf5')
'''

''' # MODEL 2 # MODEL 2 # MODEL 2 # MODEL 2 # MODEL 2 # '''

# Preprocess Input

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# Model Loading (if required)

# Compile=False Only for Predictions
model2 = load_model('models/b13_sset4_192x288_unet_incepv3_50epochs.hdf5', compile=False)
'''
# Model Definition

model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model2.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model2.load_weights('models/b13_sset4_192x288_unet_incepv3_50epochs.hdf5')
'''

''' # MODEL 3 # MODEL 3 # MODEL 3 # MODEL 3 # MODEL 3 # '''

# Preprocess Input

BACKBONE3 = 'resnet101'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# Model Loading (if required)

# Compile=False Only for Predictions
model3 = load_model('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5', compile=False)
'''
# Model Definition

model3 = sm.PSPNet(BACKBONE3, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model3.compile(optimizer='adam', loss=total_loss, metrics=metrics)
print("Model Compiled.")

# Loading Weights

model3.load_weights('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5')
'''

preprocessing_inputs = [preprocess_input1, preprocess_input2, preprocess_input3]
models = [model1, model2, model3]


''' # =================================================================================================================
# Model Testing
================================================================================================================= # '''


# TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos1.JPG'
# TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Foz do Arelho\FozDoArelho65.JPG'

# TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 1\Galapos2.JPG'
# TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 1\Galapos3.JPG'
# TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Foz do Arelho\Voo 1\FozDoArelho4.JPG'
# TEST_IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos6.JPG'

#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Samarra\Voo 3\Samarra119.JPG'
#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Samarra\Voo 2\Samarra68.JPG'
#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Samarra\Voo 2\Samarra60.JPG'
#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Samarra\Voo 1\Samarra1.JPG'

#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Foz do Arelho\Voo 2\FozDoArelho71.JPG'
#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Foz do Arelho\Voo 2\FozDoArelho65.JPG'
#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Foz do Arelho\Voo 1\FozDoArelho26.JPG'
#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Foz do Arelho\Voo 1\FozDoArelho4.JPG'

TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 2\Galapos34.JPG'
TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 2\Galapos30.JPG'
TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 2\Galapos25.JPG'
TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 1\Galapos15.JPG'
#TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 1\Galapos4.JPG'


pred_img, split_pred = biou.predict_large_image_ensemble(TEST_IMG_DIR, models, preprocessing_inputs, weights, 204, 307, 0, True, SIZE_Y, SIZE_X)
# biou.save_predicton_as_image(pred_img, 'test.png')

'''
SAVE_DIR = 'C:\\@Dissertacao\\BiologiaWork\\WM_WaterMuss\\work_with_smallset4_192x288\\test_results\\b15_sset4_ensemble_b12-b13-b14\\additional_weights-03-05-04\\'
SAVE_FILE = SAVE_DIR + 'Galapos15.png'
biou.save_predicton_as_image(pred_img, SAVE_FILE)
'''