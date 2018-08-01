from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import timeit
import random
import os

from skimage.draw import circle
from skimage.filters import sobel
from collections import deque
from sklearn.metrics import confusion_matrix
from scipy.misc import imread, imresize
from scipy.ndimage import rotate
from scipy.ndimage.filters import gaussian_filter
############################
# Neural Network Functions #
############################

# Convolution Layer
def conv(x, filter_size, num_filters, stride, weight_decay, name, padding='SAME', groups=1, trainable=True, relu=True):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[filter_size, filter_size, input_channels // groups, num_filters],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=trainable,
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', shape=[num_filters], trainable=trainable, initializer=tf.zeros_initializer())

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)
        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases

def deconv(x, filter_size, num_filters, stride, weight_decay, name, padding='SAME', relu=True):
    activation = None
    if relu:
        activation = tf.nn.relu
    return tf.layers.conv2d_transpose(x, int(num_filters), filter_size, stride, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=activation, name=name)
    
    
# Fully Connected Layer
def fc(x, num_out, weight_decay,  name, relu=True, trainable=True):
    num_in = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[num_in, num_out], 
                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                  trainable=trainable, 
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', [num_out], initializer=tf.zeros_initializer(), trainable=trainable)
        x = tf.matmul(x, weights) + biases
        if relu:
            x = tf.nn.relu(x) 
    return x

# Local Response Normalization
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def max_pool(x, filter_size, stride, name=None, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


#################################
# Training/Validation Functions #
#################################

def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output

def validate(sess, model, x_test, y_test):
    '''
    Calculates accuracy of validation set
    
    @params sess: Tensorflow Session
    @params model: Model defined from a neural network class
    @params x_test: Numpy array of validation images
    @params y_test: Numpy array of validation labels
    @params batch_size: Integer defining mini-batch size
    '''
    scores = [0] * int(y_test.shape[3]-1)
    for i in range(int(x_test.shape[0])):
        for j in range(int(y_test.shape[3]-1)):
            gt = np.argmax(y_test[i,:,:,:], 2)
            gt = create_seg(gt, j+1)
            pred = np.argmax(model.predict(sess, x_test[i:i+1])[0,:,:,:], 2)
            pred = create_seg(pred,j+1)           
            overlap = np.minimum(gt, pred)
            dice = 2*np.sum(overlap)/(np.sum(gt) + np.sum(pred) + 1)
            scores[j] = scores[j] + dice 
            
    return [score/float(x_test.shape[0]) for score in scores]

def train_print(i, j, loss, batch, batch_total, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.4} |".format(loss),
          "Data: {}/{} |".format(batch, batch_total), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")
    
def val_print(i, j, loss, acc, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.2} |".format(loss),
          "Acc: {} |".format(np.round(acc,3)), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")

    
def train(sess,
          model,
          saver,
          x_train,
          y_train,
          x_test,
          y_test,
          epochs,
          batch_size,
          auto_save_interval = 5,
          summary_writer = 0,
          train_validation = 5,
          start_step = 0,
          models_dir = None,
          model_name = None):
    '''
    Main function for training neural network model. 
    
    @params many identical to those in validate()
    @params summary_writer: Tf.summary.FileWriter used for Tensorboard variables
    @params batch_size: Integer defining mini-batch size
    @params train_validation: Integer defining how many train steps before running accuracy on training mini-batch
    '''
    losses = deque([])
    train_accs = deque([])
    step = start_step



    for i in range(epochs):

        # Shuffle indicies
        indicies = list(np.arange(x_train.shape[0]))
        np.random.shuffle(indicies)
        # Start timer
        start = timeit.default_timer()
        writer = tf.summary.FileWriter('board')
        print(int(x_train.shape[0]/batch_size))

        for j in range(int(x_train.shape[0]/batch_size)):
            # Shuffle Data
            temp_indicies = indicies[j*batch_size:(j+1)*batch_size]
            x_train_temp, y_train_temp = x_train[temp_indicies], y_train[temp_indicies]
            loss, loss_summary= model.fit_batch(sess,x_train_temp, y_train_temp)
            writer.add_summary(loss,j)
            writer.add_summary(loss_summary,j)
#             writer.add_summary(summary1,j)
#             if summary_writer:
#                 summary_writer.add_summary(loss_summary, step)
#                 summary_writer.add_summary(summary1, step)
            if len(losses) == 20:
                losses.popleft()
            losses.append(loss)
            # How often to test accuracy on training batch
            stop = timeit.default_timer()
            
            train_print(i, j, np.mean(losses), j*batch_size, x_train.shape[0], stop - start)
            step = step + 1

        # Tail case 
        if x_train.shape[0] % batch_size != 0:
            temp_indicies = indicies[(j+1)*batch_size:]
            x_train_temp, y_train_temp = x_train[temp_indicies], y_train[temp_indicies]
            loss, loss_summary= model.fit_batch(sess,x_train_temp, y_train_temp)
            writer.add_summary(loss,int(x_train.shape[0]/batch_size) + 1)
            writer.add_summary(loss_summary,int(x_train.shape[0]/batch_size) + 1)
#             writer.add_summary(summary1,int(x_train.shape[0]/batch_size) + 1)
            if len(losses) == 20:
                losses.popleft()
            losses.append(loss)
            stop = timeit.default_timer()
            train_print(i, j, np.mean(losses), j*batch_size, x_train.shape[0], stop - start)
            step = step + 1
        stop = timeit.default_timer()
        acc = validate(sess, model, x_test, y_test)
        summary = tf.Summary()
        for k in range(len(acc)):
            summary.value.add(tag="validation_acc_" + str(k), simple_value=acc[k])
#         if summary_writer:    
#             summary_writer.add_summary(summary, step) 
        wrtier.add_summary(summary, step)
        val_print(i, j, np.mean(losses), acc, stop - start)
        print()
        
        if i % auto_save_interval == 0 and i > 0:
            if models_dir and model_name:
                saver.save(sess, os.path.join(os.path.join(models_dir, model_name), model_name))
