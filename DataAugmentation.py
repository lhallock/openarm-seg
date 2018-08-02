#HOW TO RUN: Specify directory of nii files, directory of cleaned images in (1) and (2) and change degrees in (A) and angles and axes in (B)


import random
import os
import sys
import itertools
sys.path.append('src/')
import nibabel as nib
from __future__ import division, print_function, absolute_import
from sklearn.metrics import confusion_matrix
import scipy.sparse
from scipy.misc import imrotate, imresize
from scipy.ndimage import rotate
from skimage import exposure
from skimage.io import imread, imsave
from PIL import Image

arr = [45,135,225,315] #(A)
counter = '30' #used to differentiate names
for filename in os.listdir('/Users/saimandava/Desktop/allrawfillednifti/trial_12'): #(1)
    img3d = nib.load('/Users/saimandava/Desktop/allrawfillednifti/trial_12/' + filename) #(2)
    data3d = img3d.get_data()
    for a in arr:
        image_rotate=rotate(data3d, angle=a, reshape=False, axes=(1,0), order = 0) #(B)
        img3d1 = nib.Nifti1Image(image_rotate,img.affine,img.header)
        nib.save(img3d1, 'trial12_30_w3_rot' + counter + '.nii')
        counter +=('1')