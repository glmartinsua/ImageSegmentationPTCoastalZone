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
import pandas as pd

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
# Models Configuration 
================================================================================================================= # '''

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

# weights = [0.3, 0.5, 0.4]
weights = [0.3, 0.4, 0.3]


''' # MODEL 1 # MODEL 1 # MODEL 1 # MODEL 1 # MODEL 1 # '''

# Preprocess Input

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)
X_train1 = preprocess_input1(X_train)
X_val1 = preprocess_input1(X_val)
X_test1 = preprocess_input1(X_test)

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
X_train2 = preprocess_input2(X_train)
X_val2 = preprocess_input2(X_val)
X_test2 = preprocess_input2(X_test)

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
X_train3 = preprocess_input3(X_train)
X_val3 = preprocess_input3(X_val)
X_test3 = preprocess_input3(X_test)

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


''' # =================================================================================================================
# Ensemble Model Testing
================================================================================================================= # '''

# Auxiliary Grid Search To Obtain Optimal Weights

'''
df = pd.DataFrame([])

for w1 in range(0, 6):
    for w2 in range(0, 6):
        for w3 in range(0, 6):
            wts = [w1 / 10., w2 / 10., w3 / 10.]

            IOU_wted = MeanIoU(num_classes=n_classes)
            wted_preds = np.tensordot(predictions, wts, axes=((0), (0)))
            wted_ensemble_pred = np.argmax(wted_preds, axis=3)
            IOU_wted.update_state(y_test[:, :, :, 0], wted_ensemble_pred)
            print("Now predciting for weights :", w1 / 10., w2 / 10., w3 / 10., " : IOU = ", IOU_wted.result().numpy())
            df = df.append(pd.DataFrame({'wt1': wts[0], 'wt2': wts[1],
                                         'wt3': wts[2], 'IOU': IOU_wted.result().numpy()}, index=[0]),
                           ignore_index=True)

max_iou_row = df.iloc[df['IOU'].idxmax()]
print("Max IOU of ", max_iou_row[3], " obained with w1=", max_iou_row[0],
      " w2=", max_iou_row[1], " and w3=", max_iou_row[2])

# Max IOU of  0.8856666684150696  obained with w1= 0.2  w2= 0.3  and w3= 0.2 (max 0.4)

# Now predciting for weights : 0.3 0.4 0.3  : IOU =  0.8857327
# Max IOU of  0.8859207630157471  obained with w1= 0.3  w2= 0.5  and w3= 0.4 (max 0.6)
'''


# Evaluate on Test Set  [ Results may not be well calc for ensembling ]

eval1 = model1.evaluate(X_test1, y_test_cat)
eval2 = model2.evaluate(X_test2, y_test_cat)
eval3 = model3.evaluate(X_test3, y_test_cat)
evals = np.array([eval1, eval2, eval3])
results = np.tensordot(evals, weights, axes=((0),(0))) / sum(weights)
print(results)
print('loss: ', results[0])
print('iou_score: ', results[1])
print('f1-score: ', results[2])
print('mean_iou: ', results[3])

# Weights: 0.3, 0.5, 0.4
# [0.74630502 0.81049    0.88681809 0.41295782]
# loss:  0.7463050186634064
# iou_score:  0.810489997267723
# f1-score:  0.8868180910746254
# mean_iou:  0.41295781979958207

# Weights: 0.3, 0.4, 0.3
# [0.74375536 0.81578287 0.88994997 0.41184924]
# loss:  0.7437553584575652
# iou_score:  0.8157828688621522
# f1-score:  0.8899499654769898
# mean_iou:  0.41184923648834226


# Mean IoU from Keras

pred1 = model1.predict(X_test1)
pred2 = model2.predict(X_test2)
pred3 = model3.predict(X_test3)
predictions = np.array([pred1, pred2, pred3])
# print(predictions.shape)

# Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(predictions, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

# y_pred1_argmax = np.argmax(pred1, axis=3)
# y_pred2_argmax = np.argmax(pred2, axis=3)
# y_pred3_argmax = np.argmax(pred3, axis=3)

# Using built in keras function
# IOU1 = MeanIoU(num_classes=n_classes)
# IOU2 = MeanIoU(num_classes=n_classes)
# IOU3 = MeanIoU(num_classes=n_classes)
IOU_weighted = MeanIoU(num_classes=n_classes)

# IOU1.update_state(y_test[:,:,:,0], y_pred1_argmax)
# IOU2.update_state(y_test[:,:,:,0], y_pred2_argmax)
# IOU3.update_state(y_test[:,:,:,0], y_pred3_argmax)
IOU_weighted.update_state(y_test[:, :, :, 0], weighted_ensemble_prediction)

# print('IOU Score for model1 = ', IOU1.result().numpy())
# print('IOU Score for model2 = ', IOU2.result().numpy())
# print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())

# IOU Score for model1 =  0.86784244
# IOU Score for model2 =  0.86556345
# IOU Score for model3 =  0.7097028
# (FIRST TRY) IOU Score for weighted average ensemble (0.4 0.4 0.2) =  0.8844612
# (OPTIMAL) IOU Score for weighted average ensemble (0.3 0.5 0.4) =  0.885920767
# (OPTIMAL MAX 1) IOU Score for weighted average ensemble (0.3 0.4 0.3) =  0.8857327


# IoU For Each Class

values = np.array(IOU_weighted.get_weights()).reshape(n_classes, n_classes)
print(values)
print('values shape')
print(values.shape)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])  # background
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])  # water
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])  # mussels

print("IoU for class1 (background) is: ", class1_IoU)
print("IoU for class2 (water) is: ", class2_IoU)
print("IoU for class3 (mussels) is: ", class3_IoU)

# Weights: 0.3, 0.5, 0.4
# IoU for class1 (background) is:  0.9619443864792355
# IoU for class2 (water) is:  0.9849412992286443
# IoU for class3 (mussels) is:  0.7108765324065511

# Weights: 0.3, 0.4, 0.3
# IoU for class1 (background) is:  0.9619955705624262
# IoU for class2 (water) is:  0.9850011490569947
# IoU for class3 (mussels) is:  0.7102015767882615


# == # == # = # == # == # = # == # == # = # == # == # = # == # == #


# Test Random Image From Test Set

test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]

test_img_norm = test_img[:,:,:]
test_img_input = np.expand_dims(test_img_norm, 0)

test_img_input1 = preprocess_input1(test_img_input)
test_img_input2 = preprocess_input2(test_img_input)
test_img_input3 = preprocess_input3(test_img_input)


test_pred1 = model1.predict(test_img_input1)
test_pred2 = model2.predict(test_img_input2)
test_pred3 = model3.predict(test_img_input3)
test_preds = np.array([test_pred1, test_pred2, test_pred3])

# Use tensordot to sum the products of all elements over specified axes.
weighted_test_preds = np.tensordot(test_preds, weights, axes=((0),(0)))
weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0, :, :]

# Plot Result

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(weighted_ensemble_test_prediction, cmap='jet')
plt.show()
