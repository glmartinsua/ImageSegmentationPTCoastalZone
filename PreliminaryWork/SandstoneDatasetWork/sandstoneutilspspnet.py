import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def get_training_128patches():

    train_images = []
    for directory_path in glob.glob(
            "C:\\@Dissertacao\\datasets\\sandstone_data\\full_labels_for_deep_learning\\128_patches\\images\\"):
        for img_path in sorted(glob.glob(os.path.join(directory_path, "*.tif"))):
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, (144, 144))
            train_images.append(img)

    # Capture mask/label info as a list
    train_masks = []
    for directory_path in glob.glob(
            "C:\\@Dissertacao\\datasets\\sandstone_data\\full_labels_for_deep_learning\\128_patches\\masks\\"):
        for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.tif"))):
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (144, 144), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            train_masks.append(mask)

    return train_images, train_masks


def reencode_labels(train_masks):

    labelencoder = LabelEncoder()

    if train_masks.ndim != 3:
        print('NDIM != 3')
        return train_masks

    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1, 1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
    return train_masks_input


def split_train_test(train_images, train_masks, size=0.10):
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=size, random_state=0)
    return X_train, X_test, y_train, y_test


def categorical(y_masks, n_classes=4):

    if y_masks.ndim != 4:
        print('NDIM != 3')
        return y_masks

    masks_cat = to_categorical(y_masks, num_classes=n_classes)
    y_masks_cat = masks_cat.reshape((y_masks.shape[0], y_masks.shape[1], y_masks.shape[2], n_classes))
    return y_masks_cat


def load_data(n_classes=4, size_test=0.10, size_val=0.10):
    '''
    REPEAT FOR EVERY MODEL TO GET THE SAME IMAGES FOR TRAINING, VALIDATION AND TESTING
    '''

    train_images, train_masks = get_training_128patches()
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    # print(np.unique(train_masks))

    train_masks_input = reencode_labels(train_masks)

    X_train, X_test, y_train, y_test = split_train_test(train_images, train_masks_input, size_test)  # Split 10% pra teste
    X_train, X_val, y_train, y_val = split_train_test(X_train, y_train, size_val)  # Split 10% do treino para validacao

    # print("Class values in the dataset are ... ", np.unique(y_val))  # 0 is the background/few unlabeled

    y_train_cat = categorical(y_train, n_classes)
    y_val_cat = categorical(y_val, n_classes)
    y_test_cat = categorical(y_test, n_classes)

    # print("y_train_cat: ", y_train_cat.shape)
    # print("y_val_cat: ", y_val_cat.shape)
    # print("y_test_cat: ", y_test_cat.shape)

    return X_train, X_val, X_test, y_train, y_train_cat, y_val, y_val_cat, y_test, y_test_cat
