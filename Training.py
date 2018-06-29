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
local_device_protos = device_lib.list_local_devices()
print(local_device_protos)

import pipeline
import Unet



models_dir = '/media/jessica/Storage/models/u-net_v1-0'
model_name = 'new_train_test'


training_data_dir = "/home/jessica/Documents/hart-seg-ml/pipelinetest/training_data"


raw_data_lst, seg_data_lst = pipeline.load_all_data(training_data_dir)


x_train, x_val, x_test, y_train, y_val, y_test = pipeline.split_data(raw_data_lst, seg_data_lst, 60, 10, 30)


mean = 0
weight_decay = 1e-6
learning_rate = 1e-4


tf.reset_default_graph()
sess = tf.Session()
model = Unet.Unet(mean, weight_decay, learning_rate, dropout=0.5)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1)



nn.train(sess, model, saver, x_train, y_train, x_val, y_val, epochs = 16, batch_size = 1, models_dir=models_dir, model_name=model_name)


print(nn.validate(sess, model, x_test, y_test))


# pipeline.save_model(models_dir, model_name, saver, sess)

