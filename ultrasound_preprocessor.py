"""
Run with python ultrasound_preprocessor.py [directory path of .nii files] [directory path to place cleaned images]
Run requirements.txt in new virtualenv for best results.
"""

import multiprocessing as mp
import numpy as np
import os
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from PIL import Image
import sys
import scipy.sparse
import scipy.misc
from skimage.io import imread, imsave
from concurrent import futures


def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def empty_img(img):
    """
    Returns True if the image is empty -> only 0s.
    """
    return not np.count_nonzero(img)

def split_filename(filename):
    """
    Splits filename for a trial into 'trial10_30_w1', True if seg[label], False if vol[raw]
    """
    fn_lst = filename.split('_')
    if len(fn_lst) >= 4:
        trial_name = "_".join(fn_lst[:3])
        if 'seg' in fn_lst[3]:  
            return trial_name, True
        elif 'vol' in fn_lst[3]:
            return trial_name, False
    else:
        return None, None

def bounding_box(img):
    """
    Returns copy of the img bounded by the box from the image.
    """

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    box = im[rmin : rmax, cmin : cmax]
    return box

def fill(image, threshold_dist=15):
    """
    Grid fill image to pixel color that it is surrounded by [Fills in holes]
    """
    rows, cols = len(image), len(image[0])
    for u in range(rows):  # Iterate through rows
        for v in range(cols):  # Iterate through cols
            ltr_color, gtr_color, ltc_color, gtc_color = False, False, False, False
            for ltr in range(u, max(0, u-threshold_dist), -1):
                if image[ltr, v] != 0: 
                    ltr_color = image[ltr, v]
                    break
            for gtr in range(u, min(rows, u+threshold_dist)):
                if image[gtr, v] != 0: 
                    gtr_color = image[gtr, v]
                    break
            for ltc in range(v, max(0, v-threshold_dist), -1):
                if image[u, ltc] != 0: 
                    ltc_color = image[u, ltc]
                    break
            for gtc in range(v, min(cols, v+threshold_dist)):
                if image[u, gtc] != 0: 
                    gtc_color = image[u, gtc]
                    break
#             print([ltr_color, gtr_color, ltc_color, gtc_color])
            if np.all([ltr_color, gtr_color, ltc_color, gtc_color]):
                if len(set([ltr_color, gtr_color, ltc_color, gtc_color])) == 1:
                    image[u, v] = ltr_color
#               np.mean([ltr_color, gtr_color, ltc_color, gtc_color])
    return image

def one_hot_encode(L, class_labels):
    h, w = 482, 395
    try:
        encoded = np.array([list(map(class_labels.index, L.flatten()))])
    except Exception as e:
        print(e)
    L = encoded.reshape(h, w)

    Lhot = np.zeros((L.shape[0], L.shape[1], 8))
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            Lhot[i,j,L[i,j]] = 1
    return Lhot

class GenerateImages:
    def __init__(self, trial_key, raw_nii, label_nii, base_data_dir, base_img_data_dir):
        self.raw_nii_file = os.path.join(base_data_dir, raw_nii)
        self.label_nii_file = os.path.join(base_data_dir, label_nii)
        self.raw_voxel = nib.load(self.raw_nii_file).get_data()
        self.label_voxel = nib.load(self.label_nii_file).get_data()
        self.trial_img_dir = os.path.join(base_img_data_dir, trial_key)
        self.build_image_dataset()
        self.count = 0

    def parallel_helper(self, i):
        if not (empty_img(self.raw_voxel[i]) or empty_img(self.label_voxel[i])):

            file_num = str(self.count).zfill(5)
            raw_img = self.raw_voxel[i]
            save_sparse_csr(os.path.join(self.trial_img_dir, file_num + '_raw'),
                            scipy.sparse.csr_matrix(raw_img))  # saves as compressed sparse row matrix .npz of float32
            labeled_img = fill(self.label_voxel[i])  # Grid fill the labeled image
            labeled_img = labeled_img.astype(np.int16)
            imsave(os.path.join(self.trial_img_dir, file_num + '_label.png'), labeled_img)
            self.count += 1

    def build_image_dataset(self):
        if not os.path.exists(self.trial_img_dir):
            os.makedirs(self.trial_img_dir)
        with futures.ProcessPoolExecutor() as pool:
            pool.map(self.parallel_helper, range(self.raw_voxel.shape[0]))

def main(base_data_dir, base_img_data_dir):
    matched_file_dict = {}  # Dictionary of trial_key to [seg_file, vol_file]
    # base_data_dir = "/Users/kireet/ucb/HART Research/Muscle Segmentation/raw_nifti_scan"
    for filename in os.listdir(base_data_dir):
        trial_key, is_seg = split_filename(filename)
        if trial_key is not None:
            if trial_key not in matched_file_dict:
                matched_file_dict[trial_key] = [None, None]
            if is_seg:
                matched_file_dict[trial_key][1] = filename
            else:
                matched_file_dict[trial_key][0] = filename
            
    # base_img_data_dir = "/Users/kireet/ucb/HART Research/Muscle Segmentation/cleaned_images"
    # Runs the cleaning image voxel dataset -> creates cleaned 2D jpegs
    for tk, scan_lst in list(matched_file_dict.items()):
        print(tk)
        GenerateImages(tk, scan_lst[0], scan_lst[1], base_data_dir, base_img_data_dir)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        base_data_dir = sys.argv[1]
        base_img_data_dir = sys.argv[2]
        main(base_data_dir, base_img_data_dir)
    else:
        print("Run with base_nii_data_dir as arg1, and where to put image_data_dir as arg2")