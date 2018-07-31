"""
Run with python ultrasound_preprocessor.py [directory path of .nii files] [directory path to place cleaned images]
Run requirements.txt in new virtualenv for best results.
"""

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

def one_hot_encode(L, class_labels=[0, 7, 8, 9, 45, 51, 52, 53, 68]):
    """
    TODO: ensure encoding remains consistent

    2D array (image) of segmentation labels -> .npy
    # One Hot Encode the label 2d array -> .npy files with dim (h, w, len(class_labels))
    # num classes will be 8? but currently dynamically allocated based on num colors in all scans.
    """
    h, w = L.shape  # Should be 482, 395 (unless resized)
    try:
        encoded = np.array([list(map(class_labels.index, L.flatten()))])

        L = encoded.reshape(h, w)

        Lhot = np.zeros((L.shape[0], L.shape[1], len(class_labels)))
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                Lhot[i,j,L[i,j]] = 1
        return Lhot  # Should be shape (482, 395, 9)
    except Exception as e:
        logger.debug(e)

def build_image_dataset(trial_key, raw_nii, label_nii, base_data_dir, base_img_data_dir, fill_images=False):
    raw_nii_file = os.path.join(base_data_dir, raw_nii)
    label_nii_file = os.path.join(base_data_dir, label_nii)
    raw_voxel = nib.load(raw_nii_file).get_data()
    label_voxel = nib.load(label_nii_file).get_data()
    
    counter = 0
    trial_img_dir = os.path.join(base_img_data_dir, trial_key)
    if not os.path.exists(trial_img_dir):
        os.makedirs(trial_img_dir)
    raw_clean_voxel, labeled_clean_voxel = None, None
    pad = len(str(raw_voxel.shape[0]))
    for i in range(raw_voxel.shape[0]):  # shape is (1188, 482, 395)
        if empty_img(raw_voxel[i]) or empty_img(label_voxel[i]):
            continue

        file_num = str(counter).zfill(pad)

        raw_img = raw_voxel[i]
        save_sparse_csr(os.path.join(trial_img_dir, file_num + '_raw'),
                        scipy.sparse.csr_matrix(raw_img))  # saves as compressed sparse row matrix .npz of float32

        if fill_images:
            labeled_img = fill(label_voxel[i])  # Grid fill the labeled image
        else:
            labeled_img = label_voxel[i]
        labeled_img = labeled_img.astype(np.int16)
        imsave(os.path.join(trial_img_dir, file_num + '_label.png'), labeled_img)
        encoded_labeled_img = one_hot_encode(labeled_img)
        save_sparse_csr(os.path.join(trial_img_dir, file_num + '_label_enc'),
                        scipy.sparse.csr_matrix(encoded_labeled_img))


        
        counter += 1

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
        build_image_dataset(tk, scan_lst[0], scan_lst[1], base_data_dir, base_img_data_dir)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        base_data_dir = sys.argv[1]
        base_img_data_dir = sys.argv[2]
        main(base_data_dir, base_img_data_dir)
    else:
        print("Run with base_nii_data_dir as arg1, and where to put image_data_dir as arg2")