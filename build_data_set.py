import os
import numpy as np
import nibabel as nib
from nilearn import plotting
from PIL import Image


base_dir = "/Users/kireet/ucb/HART Research/Muscle Segmentation/raw_nifti_scan"
example_filename = os.path.join(base_dir, 'trial10_30_w1_seg_TRANS.nii')

img = nib.load(example_filename)
print(img.shape)

plotting.plot_stat_map(img)
plotting.show()