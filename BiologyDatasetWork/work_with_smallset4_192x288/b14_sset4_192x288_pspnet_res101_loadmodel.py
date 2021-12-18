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
X_test = preprocess_input(X_test)

# Model Loading (if required)

# Compile=False Only for Predictions
#model = load_model('models/b14_sset4_192x288_pspnet_res101_50epochs.hdf5', compile=False)

#"""
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


# Evaluate on Test Set

results = model.evaluate(X_test, y_test_cat)
print(results)
print('loss: ', results[0])
print('iou_score: ', results[1])
print('f1-score: ', results[2])
print('mean_iou: ', results[3])

# loss:  0.7973058223724365
# iou_score:  0.7048342227935791
# f1-score:  0.8237968683242798
# mean_iou:  0.5164978504180908


# Mean IoU from Keras

y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Mean IoU = 0.7097028


# IoU For Each Class

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

# IoU for class1 (background) is:  0.6741709061410344
# IoU for class2 (water) is:  0.7625695543154752
# IoU for class3 (mussels) is:  0.6923680620412793
#"""

# == # == # = # == # == # = # == # == # = # == # == # = # == # == #


# Test Random Image From Test Set

test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]

print(test_img.shape)
test_img_input = np.expand_dims(test_img, 0)
print(test_img_input.shape)

prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

# Plot Result

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

