import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


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


def show_from_png(img):
    mask = cv2.imread(img, 0)
    # X, Y
    # cv2.circle(mask, (0, 750), 50, (0, 0, 255), -1)
    print(mask.shape)
    print(np.unique(mask))
    plt.imshow(mask)
    plt.show()
    print(mask.dtype)

    # Y, X
    # print(mask[500, 2500])
    # print(mask[2500, 500])


def show_from_png_grayscale(img, save=False):
    mask = cv2.imread(img, 0)
    # X, Y
    # cv2.circle(mask, (0, 750), 50, (0, 0, 255), -1)
    print(mask.shape)
    print(np.unique(mask))
    plt.imshow(mask, cmap='gray')
    plt.show()
    print(mask.dtype)

    if save:
        plt.imsave('result_grayscale.png', mask, cmap='gray')
        # im = Image.fromarray((mask * 255).astype(np.uint8))
        # im.save('result_grayscale.png')


def show_from_png_decoded(img, save=False):
    mask = cv2.imread(img, 0)
    mask = decode_segmap(mask)
    # X, Y
    # cv2.circle(mask, (0, 750), 50, (0, 0, 255), -1)
    print(mask.shape)
    print(np.unique(mask))
    plt.imshow(mask)
    plt.show()
    print(mask.dtype)

    if save:
        # plt.imsave('result_coloured.png', mask)
        im = Image.fromarray((mask * 255).astype(np.uint8))
        im.save('result_coloured.png')


'''
Define image accordingly
'''

# base = '1'
base = 'gal43'

# img = 'mask_DSC'
img = 'mask_'

show_from_png(f'{img}{base}.png')
show_from_png_grayscale(f'{img}{base}.png', save=True)
show_from_png_decoded(f'{img}{base}.png', save=True)

