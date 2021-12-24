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
import sandstoneutilspspnet as sup


SIZE_X = 128
SIZE_Y = 128
n_classes = 4


X_train, X_val, X_test, y_train, y_train_cat, y_val, y_val_cat, y_test, y_test_cat = sup.load_data(4, 0.10, 0.10)
print("Class values in the dataset are ... ", np.unique(y_val))  # 0 is the background/few unlabeled
print("y_train_cat: ", y_train_cat.shape)
print("y_val_cat: ", y_val_cat.shape)
print("y_test_cat: ", y_test_cat.shape)


# MODEL

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

activation = 'softmax'
BACKBONE = 'resnet101'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

# define model
model = sm.PSPNet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation, input_shape=(144, 144, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

# print(model.summary())
print("Model Compiled.")

history = model.fit(X_train,
          y_train_cat,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_val, y_val_cat))


model.save('pspnet_res101_50epochs.hdf5')

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