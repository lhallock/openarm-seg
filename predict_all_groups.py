import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import sys
import itertools
sys.path.append('src/')
import nn
import process_data
import nibabel as nib
from math import floor, ceil
from sklearn.metrics import confusion_matrix
import scipy.sparse
from scipy.misc import imrotate, imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from skimage import exposure
from skimage.io import imread, imsave
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
import pipeline
import Unet
import logging
import time


over_512_configs = [("/media/jessica/Storage/SubB/prediction_sources/over_512", "/media/jessica/Storage/SubB/predictions/over_512", "/media/jessica/Storage/SubB/all_nifti")]
under_512_configs = [("/media/jessica/Storage/SubB/prediction_sources/under_512", "/media/jessica/Storage/SubB/predictions/under_512", "/media/jessica/Storage/SubB/all_nifti"),
					  ("/media/jessica/Storage/SubF/prediction_sources/under_512", "/media/jessica/Storage/SubF/predictions/under_512", "/media/jessica/Storage/SubF/all_nifti"),
					  ("/media/jessica/Storage/SubG/prediction_sources/under_512", "/media/jessica/Storage/SubG/predictions/under_512", "/media/jessica/Storage/SubG/all_nifti"),
					  ("/media/jessica/Storage/SubH/prediction_sources/under_512", "/media/jessica/Storage/SubH/predictions/under_512", "/media/jessica/Storage/SubH/all_nifti")]

# over_512_configs = []

# under_512_configs = [("/media/jessica/Storage/SubF/prediction_sources/under_512", "/media/jessica/Storage/SubF/predictions/under_512", "/media/jessica/Storage/SubF/all_nifti"),
# 					  ("/media/jessica/Storage/SubG/prediction_sources/under_512", "/media/jessica/Storage/SubG/predictions/under_512", "/media/jessica/Storage/SubG/all_nifti")]

def main():
	args = sys.argv[1:]

	models_dir = args[0] if len(args) != 0 else "/media/jessica/Storage/models/u-net_v1-0"

	group_folders = []

	for model_folder in sorted(os.listdir(models_dir))[::-1]:
		print(model_folder)
		if os.path.isdir(os.path.join(models_dir, model_folder)) and model_folder.startswith("group_"):
			group_folders.append(model_folder)

	print(group_folders)
	time.sleep(5)

	for group in group_folders:
		print(group)
		for size in [512, 1024]:
			tf.reset_default_graph()
			sess = tf.Session()
			model = Unet.Unet(0, 0.5, 0.5, h = size, w = size) # Mostly arbitrary initialization with correct size
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()

			configs = under_512_configs if size == 512 else over_512_configs

			pipeline.load_model(models_dir, group, saver, sess)

			for config in configs:
				pipeline.predict_all_segs(config[0], config[1] + "/" + group, config[2], model, sess, True)




if __name__ == '__main__':
	logger = logging.getLogger('__name__')
	stream = logging.StreamHandler(stream=sys.stdout)
	stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
	logger.handlers = []
	logger.addHandler(stream)
	logger.setLevel(logging.DEBUG)
	main()