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
import pipeline
import Unet
import logging
import argparse
import configparser
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from tensorflow.python.client import device_lib
# local_device_protos = device_lib.list_local_devices()

logger = logging.getLogger('__name__')
stream = logging.StreamHandler(stream=sys.stdout)
stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
logger.handlers = []
logger.addHandler(stream)
logger.setLevel(logging.INFO)

logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('PIL.Image').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def main():
    args = get_args()

    if args.default_models_dir or args.default_training_data_dir:
        configure_default_dirs(args.default_models_dir, args.default_training_data_dir)
        sys.exit()
    elif not os.path.isfile('trainingconfig.ini'):
        logger.info("No config file found, setting default directories.")
        configure_default_dirs(None, None)

    if not args.models_dir:
        args.models_dir = get_default_dir('models_dir')
    if not args.training_data_dir:
        args.training_data_dir = get_default_dir('training_data_dir')

    if args.debug:
        logger.setLevel(level=logging.DEBUG)

    train_model(args.models_dir,
                args.training_data_dir,
                args.epochs,
                args.batch_size,
                args.auto_save_interval,
                args.max_ckpt_keep,
                args.ckpt_n_hours,
                args.model_name)

def configure_default_dirs(default_models_dir, default_training_data_dir):
    config = configparser.ConfigParser()
    if os.path.isfile('trainingconfig.ini'):
        config.read('trainingconfig.ini')

    if default_models_dir:
        config['DEFAULT']['models_dir'] = default_models_dir
    elif 'models_dir' not in config['DEFAULT']:
        config['DEFAULT']['models_dir'] = '/media/jessica/Storage/models/u-net_v1-0'
    logger.info("Default models_dir is now %s.", config['DEFAULT']['models_dir'])

    if default_training_data_dir:
        config['DEFAULT']['training_data_dir'] = default_training_data_dir
    elif 'training_data_dir' not in config['DEFAULT']:
        config['DEFAULT']['training_data_dir'] = '/media/jessica/Storage/pipelinetest/training_data'
    logger.info("Default training_data_dir is now %s.", config['DEFAULT']['training_data_dir'])

    with open('trainingconfig.ini', 'w') as configfile:
        config.write(configfile)


def get_default_dir(directory):
    config = configparser.ConfigParser()
    config.read('trainingconfig.ini')
    return config['DEFAULT'][directory]

def get_args():
    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('model_name', action='store', nargs='?', default=None)
    parser.add_argument('--models-dir', '-md', action='store')
    parser.add_argument('--training_data_dir', '-td', action='store')
    parser.add_argument('--default-models-dir', '-dm', action='store')
    parser.add_argument('--default-training_data_dir', '-dt', action='store')
    parser.add_argument('--debug', '-de', action='store_true')
    parser.add_argument('--epochs', '-e', action='store', type=int, default=50)
    parser.add_argument('--batch_size', action='store', type=int, default=1)
    parser.add_argument('--auto_save_interval', action='store', type=int, default=5)
    parser.add_argument('--max_ckpt_keep', action='store', type=int, default=1)
    parser.add_argument('--ckpt_n_hours', action='store', type=int, default=1)

    args = parser.parse_args()

    if not args.model_name and not (args.default_models_dir or args.default_training_data_dir):
        parser.error("model_name is required when training a model.")

    return args

def train_model(models_dir, training_data_dir, num_epochs, batch_size, auto_save_interval, max_to_keep, ckpt_n_hours, model_name):
    logger.info("Fetching data.")

    raw_data_lst, seg_data_lst = pipeline.load_all_data(training_data_dir)

    train_percent = 60
    val_percent = 10
    test_percent = 30

    x_train, x_val, x_test, y_train, y_val, y_test = pipeline.split_data(raw_data_lst,
                                                                         seg_data_lst,
                                                                         train_percent,
                                                                         val_percent,
                                                                         test_percent)

    logger.info("Initializing model.")

    mean = 0
    weight_decay = 1e-6
    learning_rate = 1e-4

    tf.reset_default_graph()
    sess = tf.Session()
    model = Unet.Unet(mean, weight_decay, learning_rate, dropout=0.5)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=ckpt_n_hours)

    logger.info("Training model. Details:")
    logger.info(" * Model name: %s", model_name)
    logger.info(" * Epochs: %d", num_epochs)
    logger.info(" * Batch size: %d", batch_size)
    logger.info(" * Max checkpoints to keep: %d", max_to_keep)
    logger.info(" * Keeping checkpoint every n hours: %d", ckpt_n_hours)
    logger.info(" * Keeping checkpoint every n epochs: %d", auto_save_interval)
    logger.info(" * Model directory save destination: %s", models_dir)
    logger.info(" * Training data directory: %s", training_data_dir)

    nn.train(sess, 
             model,
             saver,
             x_train,
             y_train,
             x_val,
             y_val,
             num_epochs,
             batch_size,
             auto_save_interval,
             models_dir = models_dir,
             model_name = model_name)

    logger.info("%s", nn.validate(sess, model, x_test, y_test))

    pipeline.save_model(models_dir, model_name, saver, sess)

if __name__ == '__main__':
    main()

