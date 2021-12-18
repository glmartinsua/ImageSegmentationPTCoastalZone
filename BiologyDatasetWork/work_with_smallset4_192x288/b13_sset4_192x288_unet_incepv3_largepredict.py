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

# Model Parameters + Preprocess

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), keras.metrics.MeanIoU(num_classes=n_classes)]
activation = 'softmax'
BACKBONE = 'inceptionv3'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Model Loading (if required)

# Compile=False Only for Predictions
model = load_model('models/b13_sset4_192x288_unet_incepv3_50epochs.hdf5', compile=False)

"""
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
"""


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
TEST_IMG_DIR = 'C:\@Dissertacao\datasets\FotosBiologia\Galapos\Voo 1\Galapos4.JPG'


pred_img, split_pred = biou.predict_large_image(TEST_IMG_DIR, model, preprocess_input, 204, 307, 0, True, SIZE_Y, SIZE_X)
#biou.save_predicton_as_image(pred_img, 'test.png')


'''
SAVE_DIR = 'C:\\@Dissertacao\\BiologiaWork\\WM_WaterMuss\\work_with_smallset4_192x288\\test_results\\b13_sset4_unet_incepv3\\additional\\'
SAVE_FILE = SAVE_DIR + 'Samarra119.png'
biou.save_predicton_as_image(pred_img, SAVE_FILE)
'''