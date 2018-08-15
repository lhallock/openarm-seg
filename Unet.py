import tensorflow as tf
import nn

class Unet(object):        
    def __init__(self, mean, weight_decay, learning_rate, label_dim = 8, dropout = 0.9, h = 512, w = 512):
        self.x_train = tf.placeholder(tf.float32, [None, h, w, 1], name="x_train")
        self.y_train = tf.placeholder(tf.float32, [None, h, w, 9], name="y_train")
        self.x_test = tf.placeholder(tf.float32, [None, h, w, 1], name="x_test")
        self.y_test = tf.placeholder(tf.float32, [None, h, w, 9], name="y_test")
        
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.output= self.unet(self.x_train, mean, keep_prob=self.dropout)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.correct_pred = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y_train,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.accuracy_scalar = tf.summary.scalar("accuracy",self.accuracy)
        
        self.pred = self.unet(self.x_test, mean, reuse = True, keep_prob = 1.0)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.hist_loss = tf.summary.histogram('histogram_loss', self.loss)
#         self.summary_op = tf.summary.merge_all()

    
    
    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary= sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
#         print(type(loss))
#         print(type(_))
#         print(type(loss_summary))
        return loss, loss_summary
    
    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def unet(self, input, mean, keep_prob = 0.9, reuse = None):
        with tf.variable_scope('vgg', reuse=reuse):
            input = input - mean  # Demean
            
            pool_ = lambda x: nn.max_pool(x, 2, 2)
            
            def conv_(x, output_depth, name, padding = 'SAME', relu = True, filter_size = 3):
                result = nn.conv(x, filter_size, output_depth, 1, self.weight_decay, name=name, padding=padding, relu=relu)
                tf.summary.histogram(name, result[1])
                return result[1]
            
            def conv1_(x, output_depth, name, padding = 'SAME', relu = True, filter_size = 3):
                result = nn.conv(x, filter_size, output_depth, 1, self.weight_decay, name=name, padding=padding, relu=relu)
                tf.summary.histogram(name, result[1])
                return result[0], result[1]
            
            def deconv_(x, output_depth, name):
                result = nn.deconv(x, 2, output_depth, 2, self.weight_decay, name=name)
                tf.summary.histogram(name, result)
                return result
            
            weight, conv_1_1 = conv1_(input, 64, 'conv1_1')
            # TODO: figure out imageIndex and inputLabel so I can finish visualization,
            # TODO: FINISH the FCN implementation, start from scratch using these methods 
            batchsizeFeatures = 1
            imageIndex = 56

            inputImage = batch["data"][imageIndex:imageIndex+batchsizeFeatures]
            inputImage = inputImage/255.0
            inputImage = np.reshape(inputImage,[-1,3,512, 512])
            inputImage = np.swapaxes(inputImage,1,3)

            inputLabel = np.zeros((batchsize,9))
            inputLabel[np.arange(1),batch["labels"][imageIndex:imageIndex+batchsizeFeatures]] = 1;
            print(conv_1_1.get_shape())
            holder1 = tf.placeholder(tf.float32, [None, 512, 512, 64])
            unBias = tf.nn.relu(holder1)
            unConv = tf.nn.conv2d_transpose(unBias, weight, output_shape=[9, 512, 512, 1], strides=[1, 3, 3, 1], padding="SAME")
            activations_1_1 = conv_1_1.eval(feed_dict={self.x_train: x_train, self.y_train: y_train})
            print(activations_1_1.shape())
            for i in xrange(9):
                isolated = activations_1_1.copy()
                isolated[:,:,:,:i] = 0
                isolated[:,:,:,i+1:] = 0
                print(np.shape(isolated))
                totals = np.sum(isolated,axis=(1,2,3))
                best = np.argmin(totals,axis=0)
                print(best)
                pixelactive = unConv.eval(feed_dict={holder1: isolated})
             
                saveImage(pixelactive[best],"activ"+str(i)+".png")
#                 saveImage(inputImage[best],"activ"+str(i)+"-base.png"
            
            conv_1_2 = conv1_(conv_1_1, 64, 'conv1_2')
            holder2 = tf.placeholder(tf.float32, [None, 512, 512, 64])
            print(conv_1_2.get_shape())


            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, 128, 'conv2_1')
            print(conv_2_1.get_shape())
                   
            conv_2_2 = conv_(conv_2_1, 128, 'conv2_2')
            print(conv_2_2.get_shape())

            self.pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(self.pool_2, 256, 'conv3_1')
            conv_3_2 = conv_(conv_3_1, 256, 'conv3_2')

            pool_3 = pool_(conv_3_2)

            conv_4_1 = conv_(pool_3, 512, 'conv4_1')
            conv_4_2 = conv_(conv_4_1, 512, 'conv4_2')

            pool_4 = pool_(conv_4_2)

            conv_5_1 = conv_(pool_4, 1024, 'conv5_1')
            self.conv_5_2 = conv_(conv_5_1, 1024, 'conv5_2')
            
            pool_5 = pool_(self.conv_5_2)
            
            conv_6_1 = tf.nn.dropout(conv_(pool_5, 2048, 'conv6_1'), keep_prob)
            conv_6_2 = tf.nn.dropout(conv_(conv_6_1, 2048, 'conv6_2'), keep_prob)
            
            up_7 = tf.concat([deconv_(conv_6_2, 1024, 'up7'), self.conv_5_2], 3)  # Error here rn
            
            conv_7_1 = conv_(up_7, 1024, 'conv7_1')
            conv_7_2 = conv_(conv_7_1, 1024, 'conv7_2')

            up_8 = tf.concat([deconv_(conv_7_2, 512, 'up8'), conv_4_2], 3)
            
            conv_8_1 = conv_(up_8, 512, 'conv8_1')
            conv_8_2 = conv_(conv_8_1, 512, 'conv8_2')

            up_9 = tf.concat([deconv_(conv_8_2, 256, 'up9'), conv_3_2], 3)
            
            conv_9_1 = conv_(up_9, 256, 'conv9_1')
            conv_9_2 = conv_(conv_9_1, 256, 'conv9_2')

            up_10 = tf.concat([deconv_(conv_9_2, 128, 'up10'), conv_2_2], 3)
            
            conv_10_1 = conv_(up_10, 128, 'conv10_1')
            conv_10_2 = conv_(conv_10_1, 128, 'conv10_2')

            up_11 = tf.concat([deconv_(conv_10_2, 64, 'up11'), conv_1_2], 3)
            
            conv_11_1 = conv_(up_11, 64, 'conv11_1')
            self.conv_11_2 = conv_(conv_11_1, 64, 'conv11_2')
            
            conv_12 = conv_(self.conv_11_2, 9, 'conv12_2', filter_size = 1, relu = False)
            self.merged_summary_op = tf.summary.merge_all()
            return conv_12