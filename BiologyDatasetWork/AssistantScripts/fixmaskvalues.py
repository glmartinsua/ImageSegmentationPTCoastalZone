import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


'''
Define paths accordingly
'''

img_path = '/BiologiaWork/DatasetWaterMuss/CompleteMasks/Galapos/mask_Galapos1.Wrong.png'
save_name = 'mask_blabla.png'

imgp = Image.open(img_path)
img = np.array(imgp)

print(img.shape)

# REPLACE 1 -> 2 // 2 -> 1

new_img = np.zeros_like(img)
new_img = np.where(img == 1, 5, img)
print(np.unique(new_img))
new_img = np.where(new_img == 2, 1, new_img)
print(np.unique(new_img))
new_img = np.where(new_img == 5, 2, new_img)
print(np.unique(new_img))

plt.imshow(img)
plt.show()

plt.imshow(new_img)
plt.show()

newim = Image.fromarray(new_img)
newim.save(save_name)
