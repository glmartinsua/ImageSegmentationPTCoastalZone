"""
Variaveis do ambiente:
    SM_FRAMEWORK=tf.keras
"""

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator

# import bioworkutils_WM as biou
from BiologiaWork.WM_WaterMuss import bioworkutils_WM as biou


# DATASET DIRECTORIES
IMG_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset2_patches_204x307\images\\'
MASKS_DIR = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset2_patches_204x307\masks\\'

# Divisible by 32
SIZE_Y = 192  # 204 # height
SIZE_X = 288  # 307 # width
n_classes = 3

X_train, X_val, X_test, y_train, y_train_cat, y_val, y_val_cat, y_test, y_test_cat = biou.load_data(IMG_DIR, MASKS_DIR, n_classes, 0.10, 0.10, True, SIZE_Y, SIZE_X)
print("Class values in the dataset are ... ", np.unique(y_val))  # 0 is the background/few unlabeled
print("y_train_cat: ", y_train_cat.shape)
print("y_val_cat: ", y_val_cat.shape)
print("y_test_cat: ", y_test_cat.shape)

# MODEL

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

activation = 'softmax'
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
# X_train = preprocess_input(X_train)  # Only preprocess after augmentation
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)


# AUGMENTATION
# https://github.com/bnsreenu/python_for_microscopists/blob/master/216_mito_unet_12_training_images_V1.0.py

#           +-> training set ---> data augmentation --+
#           |                                         |
#           |                                         +-> model training --+
#           |                                         |                    |
# all data -+-> validation set -----------------------+                    |
#           |                                                              +-> model testing
#           |                                                              |
#           |                                                              |
#           +-> test set --------------------------------------------------+


SEED = 0
BATCH_SIZE = 16
STEPS_PER_EPOCH = (len(X_train))//BATCH_SIZE
print(STEPS_PER_EPOCH)


# INPUT

img_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(X_train, augment=True, seed=SEED)
X_train_generator = image_data_generator.flow(X_train, seed=SEED, batch_size=BATCH_SIZE)


# TARGET

mask_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect',
                          preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype))  # Binarize the output again.

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y_train_cat, augment=True, seed=SEED)
# y_train_generator = mask_data_generator.flow(y_train_cat, seed=SEED, batch_size=BATCH_SIZE)
y_train_generator = mask_data_generator.flow(y_train, seed=SEED, batch_size=BATCH_SIZE)


def get_xy_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


def get_xy_generator_with_preprocess(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img = preprocess_input(img)
        yield (img, mask)


def get_xy_generator_with_preprocess_and_cat(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img = preprocess_input(img)
        mask = biou.categorical(mask, n_classes)
        yield (img, mask)


# train_datagen = get_xy_generator_with_preprocess(X_train_generator, y_train_generator)
train_datagen = get_xy_generator_with_preprocess_and_cat(X_train_generator, y_train_generator)

# x1 = next(train_datagen)
# x2 = next(train_datagen)


# custom loss with weights
CLASS_0_WEIGHT = 0.1
CLASS_1_WEIGHT = 0.2
CLASS_2_WEIGHT = 0.7

# Segmentation models losses
dice_loss = sm.losses.DiceLoss(class_weights=np.array([CLASS_0_WEIGHT, CLASS_1_WEIGHT, CLASS_2_WEIGHT]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(SIZE_Y, SIZE_X, 3), classes=n_classes, activation=activation)
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

# print(model.summary())
print("Model Compiled.")


history = model.fit(train_datagen,
                    validation_data=(X_val, y_val_cat),
                    steps_per_epoch=STEPS_PER_EPOCH,
                    # validation_steps=STEPS_PER_EPOCH,
                    epochs=50)

model.save('new_b5_sset2_aug_totalloss_192x288_unet_res34_50epochs.hdf5')


'''
results = model.evaluate(X_test, y_test_cat)
print(results)
print('loss: ', results[0])
print('iou_score: ', results[1])
print('f1-score: ', results[2])
'''


# plot the training and validation accuracy and loss at each epoch
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