import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np


def save_as_png(path, base):
    for mask_path in sorted(glob.glob(path)):
        mask = cv2.imread(mask_path, 0)
        arr = np.array(mask)
        print(arr.shape)
        print(np.unique(arr))
        plt.imshow(mask)
        plt.show()
        cv2.imwrite(f'mask_{base}.png', mask)


def show_from_png(img):
    mask = cv2.imread(img, 0)
    arr = np.array(mask)
    print(arr.shape)
    print(np.unique(arr))
    plt.imshow(mask)
    plt.show()


'''
Define path accordingly
'''

# path = 'C:\@Dissertacao\BiologiaWork\MaskConverter\DSC1_test.ome.tiff'
path = 'C:\@Dissertacao\BiologiaWork\@auxiliares\MaskConverter\Galapos43_Water_Mussels.ome.tiff'

basename = os.path.basename(path).split('_')[0]
print(basename)

save_as_png(path, basename)
show_from_png(f'mask_{basename}.png')