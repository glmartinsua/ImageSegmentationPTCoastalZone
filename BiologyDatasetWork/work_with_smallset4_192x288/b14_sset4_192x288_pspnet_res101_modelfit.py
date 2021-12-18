"""
Variaveis do ambiente:
    SM_FRAMEWORK=tf.keras
"""

import segmentation_models as sm
import numpy as np
from matplotlib import pyplot as plt

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

# Class Frequency Verification

unique_elements, counts_elements = np.unique(ar=y_train, return_counts=True)
class_frequency = counts_elements / np.sum(counts_elements)

print("Number of elements: ", counts_elements)
print("Class frequency: ", class_frequency)
# Number of elements:  [32927523 36814996  1921097]
# Class frequency: [0.45947337 0.51371949 0.02680715]


''' # =================================================================================================================
# Model Compilation 
================================================================================================================= # '''


# Model Parameters

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
activation = 'softmax'
BACKBONE = 'resnet101'

# Preprocess Input

preprocess_input = sm.get_preprocessing(BACKBONE)
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

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

# print(model.summary())
print("Model Compiled.")


''' # =================================================================================================================
# Model Training 
================================================================================================================= # '''


# Model Fit + Save

history = model.fit(X_train,
          y_train_cat,
          batch_size=16,
          epochs=50,
          verbose=1,
          validation_data=(X_val, y_val_cat))

model.save('new_sset4_192x288_pspnet_res101_50epochs.hdf5')

# Plot Results

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

