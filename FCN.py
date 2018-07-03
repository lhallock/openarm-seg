import tensorflow as tf
import nn
import os
import TensorflowUtils as utils
import numpy as np

#TODO: figure out what REUSE and MEAN is



class FCN(object):
    def __init__(self, mean, weight_decay, learning_rate, label_dim=8, dropout=0.9, h=512, w=512):
        self.NUM_CLASSES = 2 # TODO: HARDCODE THIS
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.x_test = None

        self.output = self.fcn(self.x_train, mean, keep_prob=self.dropout)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.pred = self.fcn(self.x_test, mean, reuse=True, keep_prob=.8)
        self.loss_summary = tf.summary.scalar('loss', self.loss)

    def fcn(self, input, mean, keep_prob = 0.9, reuse = None):
        input = input - mean  # Demean

        pool_ = lambda x: nn.max_pool(x, 2, 2)
        conv_ = lambda x, output_depth, name, padding='SAME', relu=True, filter_size=3: nn.conv(x, filter_size,
                                                                                                output_depth, 1,
                                                                                                self.weight_decay,
                                                                                                name=name,
                                                                                                padding=padding,
                                                                                                relu=relu)
        deconv_ = lambda x, output_depth, name: nn.deconv(x, 2, output_depth, 2, self.weight_decay, name=name)

        # construct fully convolutional layers

        conv_1_1 = conv_(input, 64, 'conv1_1')
        conv_1_2 = conv_(conv_1_1, 64, 'conv1_2')

        pool_1 = pool_(conv_1_2)

        conv_2_1 = conv_(pool_1, 128, 'conv2_1')
        conv_2_2 = conv_(conv_2_1, 128, 'conv2_2')

        pool_2 = pool_(conv_2_2)

        conv_3_1 = conv_(pool_2, 256, 'conv3_1')
        conv_3_2 = conv_(conv_3_1, 256, 'conv3_2')

        pool_3 = pool_(conv_3_2)

        conv_4_1 = conv_(pool_3, 512, 'conv4_1')
        conv_4_2 = conv_(conv_4_1, 512, 'conv4_2')

        pool_4 = pool_(conv_4_2)

        conv_5_1 = conv_(pool_4, 4096, 'conv5_1')
        conv_5_2 = conv_(conv_5_1, 4096, 'conv5_2')

        pool_5 = pool_(conv_5_2)

        # after pool 5, 4096 depth and start decoding, 1x1 convolution and compress depth to num of classes

        w6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv_6 = utils.conv2d_basic(pool_5, w6, b6)
        relu_6 = tf.nn.relu(conv_6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu_6, keep_prob=keep_prob)

        w7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")  # 1X1 Convloution
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, w7, b7)  # 1X1 Convloution
        relu7 = tf.nn.relu(conv7, name="relu7")
        # if FLAGS.debug: utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        w8 = utils.weight_variable([1, 1, 4096, self.NUM_CLASSES],
                                   name="W8")
        b8 = utils.bias_variable([self.NUM_CLASSES], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, w8, b8)

        # upscale image back to orgingal size

        deconv_shape1 = pool_4.get_shape()  # Set the output shape for the the transpose convolution output take only the depth since the transpose convolution will have to have the same depth for output
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, self.NUM_CLASSES],
                                     name="W_t1")  # Deconvolution/transpose in size 4X4 note that the output shape is of  depth NUM_OF_CLASSES this is not necessary in will need to be fixed if you only have 2 catagories
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(
            pool_4))  # Use strided convolution to double layer size (depth is the depth of pool4 for the later element wise addition
        fuse_1 = tf.add(conv_t1, pool_4, name="fuse_1")

        deconv_shape2 = pool_3.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool_3))
        fuse_2 = tf.add(conv_t2, pool_3, name="fuse_2")

        shape = tf.shape(self.label_dim) # change to input dimensions
        W_t3 = utils.weight_variable([16, 16, self.NUM_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([self.NUM_CLASSES], name="b_t3")

        prob = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3,
                                                   output_shape=[shape[0], shape[1], shape[2], self.NUM_CLASSES], stride=8)
        pred = tf.argmax(prob, dimension=3, name="prediction")
        return prob

    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction


# TODO : Write the prediction function, don't need the fit_batch method