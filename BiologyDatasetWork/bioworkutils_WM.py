import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.metrics import MeanIoU


''' #
# IMAGE VISUALIZATION
# '''


def get_bio_labels():
    return np.asarray(
        [
            [0, 0, 0],        # background
            [51, 153, 255],   # water - rgb(51, 153, 255)
            [204, 102, 0],    # mussels - rgb(204, 102, 0)
        ]
    )


def decode_segmap(label_mask, plot=False):
    """
    Decode segmentation class labels into a color image
    """
    n_classes = 3
    label_colours = get_bio_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(30, 20))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


''' #
# OTHER FUNCS
# '''


def get_training_patches(img_dir, masks_dir, resize=False, size_y=204, size_x=307):
    """ get training patches (image + mask) from given folders """

    if resize:

        train_images = []
        for directory_path in glob.glob(img_dir):
            for img_path in sorted(glob.glob(os.path.join(directory_path, "*.JPG"))):
                img = Image.open(img_path)
                img = img.resize((size_x, size_y), Image.ANTIALIAS)  # width, height
                img = np.array(img)
                train_images.append(img)

        # Capture mask/label info as a list
        train_masks = []
        for directory_path in glob.glob(masks_dir):
            for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
                mask = cv2.imread(mask_path, 0)
                mask = cv2.resize(mask, (size_x, size_y), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
                train_masks.append(mask)

        return train_images, train_masks

    else:

        train_images = []
        for directory_path in glob.glob(img_dir):
            for img_path in sorted(glob.glob(os.path.join(directory_path, "*.JPG"))):
                img = Image.open(img_path)
                # img = img.resize((288, 192), Image.ANTIALIAS)  # width, height
                img = np.array(img)
                train_images.append(img)

        # Capture mask/label info as a list
        train_masks = []
        for directory_path in glob.glob(masks_dir):
            for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
                mask = cv2.imread(mask_path, 0)
                # mask = cv2.resize(mask, (288, 192), interpolation=cv2.INTER_NEAREST)  # Otherwise ground truth changes due to interpolation
                train_masks.append(mask)

        return train_images, train_masks


def reencode_labels(train_masks):
    """Make sure labels are correctly given from 0-n_classes"""

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
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=size, random_state=0)  # stratify=train_masks -> erro
    return X_train, X_test, y_train, y_test


def categorical(y_masks, n_classes=3):

    if y_masks.ndim != 4:
        print('NDIM != 4')
        return y_masks

    masks_cat = to_categorical(y_masks, num_classes=n_classes)
    y_masks_cat = masks_cat.reshape((y_masks.shape[0], y_masks.shape[1], y_masks.shape[2], n_classes))
    return y_masks_cat


def load_data(img_dir, masks_dir, n_classes=3, size_test=0.10, size_val=0.10, resize=False, size_y=204, size_x=307):
    '''
    REPEAT FOR EVERY MODEL TO GET THE SAME IMAGES FOR TRAINING, VALIDATION AND TESTING
    '''

    train_images, train_masks = get_training_patches(img_dir, masks_dir, resize, size_y, size_x)
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    # print(np.unique(train_masks))

    train_masks_input = reencode_labels(train_masks)

    # print(train_masks.shape)
    # print(train_masks_input.shape)

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


# ==== #

def start_points(size, split_size, overlap=0):
    """
    Get starting points for each patch accoring to overlap
    """
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def predict_large_image_without_preprocess(img_dir, model, split_y=204, split_x=307, overlap=0.0, resize=False, size_y=204, size_x=307):
    imgp = Image.open(img_dir)
    origin_img = np.array(imgp)
    try:
        img_h, img_w, _ = origin_img.shape
    except ValueError:
        img_h, img_w = origin_img.shape

    print(f'Original image shape: {origin_img.shape}')

    # Original Image
    plt.imshow(imgp)
    plt.show()

    X_points = start_points(img_w, split_x, overlap)
    Y_points = start_points(img_h, split_y, overlap)

    print(f'len(X_points): {len(X_points)}')
    print(f'len(Y_points): {len(Y_points)}')

    splitted_images = []
    for i in Y_points:
        for j in X_points:
            split = origin_img[i:i + split_y, j:j + split_x]
            splitted_images.append(split)

    print(f'Total patches: {len(splitted_images)}')
    print(f'Each patch has shape: {splitted_images[0].shape}')

    # PREDICTION
    prediction_splitted = []

    if resize:
        for img in splitted_images:
            # print('NEW IMAGE')
            # print(img.shape)
            img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            # print(img.shape)
            img = np.expand_dims(img, 0)
            prediction = (model.predict(img))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]
            # print(predicted_img.shape)
            predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            # print(predicted_img.shape)
            prediction_splitted.append(predicted_img)
    else:
        for img in splitted_images:
            # img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            img = np.expand_dims(img, 0)
            prediction = (model.predict(img))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]
            # predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            prediction_splitted.append(predicted_img)

    # print(splitted_images)
    # print(prediction_splitted)

    print(f'Total predicted patches: {len(prediction_splitted)}')
    print(f'Each predicted patch has shape: {prediction_splitted[0].shape}')

    # final_image = np.zeros_like(origin_img)
    final_image = np.zeros((img_h, img_w), dtype=int)
    index = 0
    for i in Y_points:
        for j in X_points:
            final_image[i:i + split_y, j:j + split_x] = prediction_splitted[index]
            index += 1

    plt.imshow(final_image)
    plt.show()

    return final_image, prediction_splitted


def predict_large_image(img_dir, model, preprocessing_func, split_y=204, split_x=307, overlap=0.0, resize=False, size_y=204, size_x=307):
    imgp = Image.open(img_dir)
    origin_img = np.array(imgp)
    try:
        img_h, img_w, _ = origin_img.shape
    except ValueError:
        img_h, img_w = origin_img.shape

    print(f'Original image shape: {origin_img.shape}')

    # Original Image
    plt.imshow(imgp)
    plt.show()

    X_points = start_points(img_w, split_x, overlap)
    Y_points = start_points(img_h, split_y, overlap)

    print(f'len(X_points): {len(X_points)}')
    print(f'len(Y_points): {len(Y_points)}')

    splitted_images = []
    for i in Y_points:
        for j in X_points:
            split = origin_img[i:i + split_y, j:j + split_x]
            splitted_images.append(split)

    print(f'Total patches: {len(splitted_images)}')
    print(f'Each patch has shape: {splitted_images[0].shape}')

    # PREDICTION
    prediction_splitted = []

    if resize:
        for img in splitted_images:
            # print('NEW IMAGE')
            # print(img.shape)
            img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            # print(img.shape)
            img = preprocessing_func(img)
            img = np.expand_dims(img, 0)
            prediction = (model.predict(img))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]
            # print(predicted_img.shape)
            predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            # print(predicted_img.shape)
            prediction_splitted.append(predicted_img)
    else:
        for img in splitted_images:
            # img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            img = preprocessing_func(img)
            img = np.expand_dims(img, 0)
            prediction = (model.predict(img))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]
            # predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            prediction_splitted.append(predicted_img)

    # print(splitted_images)
    # print(prediction_splitted)

    print(f'Total predicted patches: {len(prediction_splitted)}')
    print(f'Each predicted patch has shape: {prediction_splitted[0].shape}')

    # final_image = np.zeros_like(origin_img)
    final_image = np.zeros((img_h, img_w), dtype=int)
    index = 0
    for i in Y_points:
        for j in X_points:
            final_image[i:i + split_y, j:j + split_x] = prediction_splitted[index]
            index += 1

    plt.imshow(final_image)
    plt.show()

    return final_image, prediction_splitted


def save_predicton_as_image(predicton, save_dir='saved_pred.png'):
    img = decode_segmap(predicton)
    im = Image.fromarray((img * 255).astype(np.uint8))
    im.save(save_dir)


def predict_and_evaluate_large_image(img_dir, mask_dir, model, preprocessing_func, n_classes=3, split_y=204, split_x=307, overlap=0.0, resize=False, size_y=204, size_x=307):
    """ Predict large image and provide evaluation results"""

    mask = cv2.imread(mask_dir, 0)
    imgp = Image.open(img_dir)
    origin_img = np.array(imgp)
    try:
        img_h, img_w, _ = origin_img.shape
    except ValueError:
        img_h, img_w = origin_img.shape

    print(f'Original image shape: {origin_img.shape}')

    # Original Image
    plt.imshow(imgp)
    plt.show()

    # Original Mask
    plt.imshow(mask)
    plt.show()

    X_points = start_points(img_w, split_x, overlap)
    Y_points = start_points(img_h, split_y, overlap)

    print(f'len(X_points): {len(X_points)}')
    print(f'len(Y_points): {len(Y_points)}')

    splitted_images = []
    for i in Y_points:
        for j in X_points:
            split = origin_img[i:i + split_y, j:j + split_x]
            splitted_images.append(split)

    splitted_masks = []
    for i in Y_points:
        for j in X_points:
            split = mask[i:i + split_y, j:j + split_x]
            splitted_masks.append(split)

    print(f'Total patches: {len(splitted_images)}')
    print(f'Each patch has shape: {splitted_images[0].shape}')

    # PREDICTION
    prediction_splitted = []
    # Used to evaluate
    test_images = []
    test_masks = []

    if resize:
        for img in splitted_images:
            # print('NEW IMAGE')
            # print(img.shape)
            img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_images.append(img)
            # print(img.shape)
            img = preprocessing_func(img)
            img = np.expand_dims(img, 0)
            prediction = (model.predict(img))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]
            # print(predicted_img.shape)
            predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            # print(predicted_img.shape)
            prediction_splitted.append(predicted_img)

        for mask in splitted_masks:
            mask = cv2.resize(mask, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_masks.append(mask)

    else:
        for img in splitted_images:
            # img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_images.append(img)
            img = preprocessing_func(img)
            img = np.expand_dims(img, 0)
            prediction = (model.predict(img))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]
            # predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            prediction_splitted.append(predicted_img)

        for mask in splitted_masks:
            # mask = cv2.resize(mask, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_masks.append(mask)

    # print(splitted_images)
    # print(prediction_splitted)

    print(f'Total predicted patches: {len(prediction_splitted)}')
    print(f'Each predicted patch has shape: {prediction_splitted[0].shape}')

    # final_image = np.zeros_like(origin_img)
    final_image = np.zeros((img_h, img_w), dtype=int)
    index = 0
    for i in Y_points:
        for j in X_points:
            final_image[i:i + split_y, j:j + split_x] = prediction_splitted[index]
            index += 1

    plt.imshow(final_image)
    plt.show()

    # SET TEST SET TO EVALUATE

    test_images = np.array(test_images)
    test_images = preprocessing_func(test_images)
    test_masks = np.array(test_masks)
    test_masks = reencode_labels(test_masks)
    test_masks_cat = categorical(test_masks, n_classes)

    results = model.evaluate(test_images, test_masks_cat)
    print(f'results (in metrics order): {results}\n')

    y_pred = model.predict(test_images)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(test_masks[:, :, :, 0], y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    # To calculate I0U for each class... (Show if classes -> 3)
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    if n_classes == 3:
        print(values)
        print('values shape')
        print(values.shape)
        class1_IoU = values[0, 0] / (values[0, 0] + values[0, 1] + values[0, 2] + values[1, 0] + values[2, 0])  # background
        class2_IoU = values[1, 1] / (values[1, 1] + values[1, 0] + values[1, 2] + values[0, 1] + values[2, 1])  # water
        class3_IoU = values[2, 2] / (values[2, 2] + values[2, 0] + values[2, 1] + values[0, 2] + values[1, 2])  # mussels
        print("IoU for class1 (0) is: ", class1_IoU)
        print("IoU for class2 (1) is: ", class2_IoU)
        print("IoU for class3 (2) is: ", class3_IoU)

    return final_image, prediction_splitted, results, values


# == # == #

# todo: Ensemble

def predict_large_image_ensemble(img_dir, models, preprocessing_funcs, weights, split_y=204, split_x=307, overlap=0.0, resize=False, size_y=204, size_x=307):
    imgp = Image.open(img_dir)
    origin_img = np.array(imgp)
    try:
        img_h, img_w, _ = origin_img.shape
    except ValueError:
        img_h, img_w = origin_img.shape

    print(f'Original image shape: {origin_img.shape}')

    # Original Image
    plt.imshow(imgp)
    plt.show()

    X_points = start_points(img_w, split_x, overlap)
    Y_points = start_points(img_h, split_y, overlap)

    print(f'len(X_points): {len(X_points)}')
    print(f'len(Y_points): {len(Y_points)}')

    splitted_images = []
    for i in Y_points:
        for j in X_points:
            split = origin_img[i:i + split_y, j:j + split_x]
            splitted_images.append(split)

    print(f'Total patches: {len(splitted_images)}')
    print(f'Each patch has shape: {splitted_images[0].shape}')

    # PREDICTION
    prediction_splitted = []

    if resize:
        for img in splitted_images:
            # print('NEW IMAGE')
            # print(img.shape)
            img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            # print(img.shape)
            img = img[:, :, :]
            img = np.expand_dims(img, 0)

            preds = []
            i = 0
            for model in models:
                img = preprocessing_funcs[i](img)
                prediction = model.predict(img)
                preds.append(prediction)
                i += 1

            preds = np.array(preds)
            weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
            predicted_img = np.argmax(weighted_preds, axis=3)[0, :, :]
            # print(predicted_img.shape)
            predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            # print(predicted_img.shape)
            prediction_splitted.append(predicted_img)
    else:
        for img in splitted_images:
            # img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            img = img[:, :, :]
            img = np.expand_dims(img, 0)

            preds = []
            i = 0
            for model in models:
                img = preprocessing_funcs[i](img)
                prediction = model.predict(img)
                preds.append(prediction)
                i += 1

            preds = np.array(preds)
            weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
            predicted_img = np.argmax(weighted_preds, axis=3)[0, :, :]
            # predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            prediction_splitted.append(predicted_img)

    # print(splitted_images)
    # print(prediction_splitted)

    print(f'Total predicted patches: {len(prediction_splitted)}')
    print(f'Each predicted patch has shape: {prediction_splitted[0].shape}')

    # final_image = np.zeros_like(origin_img)
    final_image = np.zeros((img_h, img_w), dtype=int)
    index = 0
    for i in Y_points:
        for j in X_points:
            final_image[i:i + split_y, j:j + split_x] = prediction_splitted[index]
            index += 1

    plt.imshow(final_image)
    plt.show()

    return final_image, prediction_splitted


def predict_and_evaluate_large_image_ensemble(img_dir, mask_dir, models, preprocessing_funcs, weights, n_classes=3, split_y=204, split_x=307, overlap=0.0, resize=False, size_y=204, size_x=307):
    """ Predict large image and provide evaluation results"""

    mask = cv2.imread(mask_dir, 0)
    imgp = Image.open(img_dir)
    origin_img = np.array(imgp)
    try:
        img_h, img_w, _ = origin_img.shape
    except ValueError:
        img_h, img_w = origin_img.shape

    print(f'Original image shape: {origin_img.shape}')

    # Original Image
    plt.imshow(imgp)
    plt.show()

    # Original Mask
    plt.imshow(mask)
    plt.show()

    X_points = start_points(img_w, split_x, overlap)
    Y_points = start_points(img_h, split_y, overlap)

    print(f'len(X_points): {len(X_points)}')
    print(f'len(Y_points): {len(Y_points)}')

    splitted_images = []
    for i in Y_points:
        for j in X_points:
            split = origin_img[i:i + split_y, j:j + split_x]
            splitted_images.append(split)

    splitted_masks = []
    for i in Y_points:
        for j in X_points:
            split = mask[i:i + split_y, j:j + split_x]
            splitted_masks.append(split)

    print(f'Total patches: {len(splitted_images)}')
    print(f'Each patch has shape: {splitted_images[0].shape}')

    # PREDICTION
    prediction_splitted = []
    # Used to evaluate
    test_images = []
    test_masks = []

    if resize:
        for img in splitted_images:
            # print('NEW IMAGE')
            # print(img.shape)
            img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_images.append(img)
            # print(img.shape)
            img = img[:, :, :]
            img = np.expand_dims(img, 0)

            preds = []
            i = 0
            for model in models:
                img = preprocessing_funcs[i](img)
                prediction = model.predict(img)
                preds.append(prediction)
                i += 1

            preds = np.array(preds)
            weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
            predicted_img = np.argmax(weighted_preds, axis=3)[0, :, :]
            # print(predicted_img.shape)
            predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            # print(predicted_img.shape)
            prediction_splitted.append(predicted_img)

        for mask in splitted_masks:
            mask = cv2.resize(mask, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_masks.append(mask)

    else:
        for img in splitted_images:
            # img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_images.append(img)
            img = img[:, :, :]
            img = np.expand_dims(img, 0)

            preds = []
            i = 0
            for model in models:
                img = preprocessing_funcs[i](img)
                prediction = model.predict(img)
                preds.append(prediction)
                i += 1

            preds = np.array(preds)
            weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
            predicted_img = np.argmax(weighted_preds, axis=3)[0, :, :]
            # predicted_img = cv2.resize(predicted_img, (split_x, split_y), interpolation=cv2.INTER_NEAREST)
            prediction_splitted.append(predicted_img)

        for mask in splitted_masks:
            # mask = cv2.resize(mask, (size_x, size_y), interpolation=cv2.INTER_NEAREST)
            test_masks.append(mask)

    # print(splitted_images)
    # print(prediction_splitted)

    print(f'Total predicted patches: {len(prediction_splitted)}')
    print(f'Each predicted patch has shape: {prediction_splitted[0].shape}')

    # final_image = np.zeros_like(origin_img)
    final_image = np.zeros((img_h, img_w), dtype=int)
    index = 0
    for i in Y_points:
        for j in X_points:
            final_image[i:i + split_y, j:j + split_x] = prediction_splitted[index]
            index += 1

    plt.imshow(final_image)
    plt.show()

    # SET TEST SET TO EVALUATE

    test_masks = np.array(test_masks)
    test_masks = reencode_labels(test_masks)
    test_masks_cat = categorical(test_masks, n_classes)

    test_images = np.array(test_images)
    # test_images = preprocessing_func(test_images)

    test_preds = []
    test_evals = []
    i = 0
    for model in models:
        test_im = preprocessing_funcs[i](test_images)
        prediction = model.predict(test_im)
        evaluation = model.evaluate(test_im, test_masks_cat)
        test_preds.append(prediction)
        test_evals.append(evaluation)
        i += 1

    test_preds = np.array(test_preds)
    test_evals = np.array(test_evals)

    weighted_tevals = np.tensordot(test_evals, weights, axes=((0), (0))) / sum(weights)
    print(f'results (in metrics order): {weighted_tevals}\n')

    weighted_tpreds = np.tensordot(test_preds, weights, axes=((0), (0)))
    weighted_ensemble_tpreds = np.argmax(weighted_tpreds, axis=3)

    IOU_weighted = MeanIoU(num_classes=n_classes)
    IOU_weighted.update_state(test_masks[:, :, :, 0], weighted_ensemble_tpreds)
    print("Mean IoU =", IOU_weighted.result().numpy())

    # To calculate I0U for each class... (Show if classes -> 3)
    values = np.array(IOU_weighted.get_weights()).reshape(n_classes, n_classes)
    if n_classes == 3:
        print(values)
        print('values shape')
        print(values.shape)
        class1_IoU = values[0, 0] / (values[0, 0] + values[0, 1] + values[0, 2] + values[1, 0] + values[2, 0])  # background
        class2_IoU = values[1, 1] / (values[1, 1] + values[1, 0] + values[1, 2] + values[0, 1] + values[2, 1])  # water
        class3_IoU = values[2, 2] / (values[2, 2] + values[2, 0] + values[2, 1] + values[0, 2] + values[1, 2])  # mussels
        print("IoU for class1 (0) is: ", class1_IoU)
        print("IoU for class2 (1) is: ", class2_IoU)
        print("IoU for class3 (2) is: ", class3_IoU)

    return final_image, prediction_splitted, weighted_tevals, values

