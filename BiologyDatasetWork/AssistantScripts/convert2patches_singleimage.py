import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# AUX FUNC
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


def split_and_save(img_path, img_dir_out, split_width=307, split_height=204, overlap=0.0, verify_total=False):

    filename = os.path.basename(img_path)
    img_name, out_type = filename.split('.')

    imgp = Image.open(img_path)
    img = np.array(imgp)

    try:
        img_h, img_w, _ = img.shape
    except ValueError:
        img_h, img_w = img.shape

    print(f'Original image shape: {img.shape}')

    # Original Image
    plt.imshow(imgp)
    plt.show()

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    print(f'len(X_points): {len(X_points)}')
    print(f'len(Y_points): {len(Y_points)}')

    splitted_images = []
    for i in Y_points:
        for j in X_points:
            split = img[i:i + split_height, j:j + split_width]
            splitted_images.append(split)
            # Save To File
            out = os.path.join(img_dir_out, f'{img_name}_{i}_{j}.{out_type}')
            im = Image.fromarray(split)
            im.save(out)

    print(f'Total patches: {len(splitted_images)}')
    print(f'Each patch has shape: {splitted_images[0].shape}')

    # Glue Patches Back Together To Verify
    if verify_total:
        final_image = np.zeros_like(img)
        index = 0
        for i in Y_points:
            for j in X_points:
                final_image[i:i + split_height, j:j + split_width] = splitted_images[index]
                index += 1

        plt.imshow(final_image)
        plt.show()

    return splitted_images


'''
Define parameters and paths accordingly
'''


# 4912 x 3264 (MDC = 16)
# -> 4912/16 = 307
# -> 3264/16 = 204

SPLIT_WIDTH = 307
SPLIT_HEIGHT = 204
OVERLAP = 0.20  # 0

"""
# FozDoArelho65

img_path = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Foz do Arelho\FozDoArelho65.JPG'
img_dir_out = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset1_patches_204x307_FozDoArelho65_Galapos1\images'

mask_path = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Foz do Arelho\mask_FozDoArelho65.png'
mask_dir_out = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallset1_patches_204x307_FozDoArelho65_Galapos1\masks'

"""

# GalaposX

img_path = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteImages\Galapos\Galapos36.JPG'
img_dir_out = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallsetX_patches_204x307_GalaposX4\images'

mask_path = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\CompleteMasks\Galapos\mask_Galapos36.png'
mask_dir_out = 'C:\@Dissertacao\BiologiaWork\WM_WaterMuss\DatasetWM\smallsetX_patches_204x307_GalaposX4\masks'


img_patches = split_and_save(img_path, img_dir_out, split_width=SPLIT_WIDTH, split_height=SPLIT_HEIGHT, overlap=OVERLAP, verify_total=True)
mask_patches = split_and_save(mask_path, mask_dir_out, split_width=SPLIT_WIDTH, split_height=SPLIT_HEIGHT, overlap=OVERLAP, verify_total=True)


'''
# Verify
print(img_patches[0].shape)
print(mask_patches[0].shape)

plt.imshow(img_patches[0])
plt.show()
plt.imshow(mask_patches[0])
plt.show()

plt.imshow(img_patches[1])
plt.show()
plt.imshow(mask_patches[1])
plt.show()

plt.imshow(img_patches[2])
plt.show()
plt.imshow(mask_patches[2])
plt.show()

plt.imshow(img_patches[3])
plt.show()
plt.imshow(mask_patches[3])
plt.show()
'''

