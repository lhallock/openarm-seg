
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print("imported")
import random
import os
import sys
import itertools
sys.path.append('src/')
import nn
import process_data
import nibabel as nib
# import cv2

from __future__ import division, print_function, absolute_import
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


# # Modeling Data Setup

# ### Load Data from File

# In[2]:


def one_hot_encode(L, class_labels):
    """
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
            print(e)

def uncode_one_hot(npy_file):
    """
    .npy file -> JPEG
    """
    pass

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
#         if image.ndim == 2:
#             plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
def load_sparse_csr(filename):
    # Sparse matrix reading function to read our raw .npz files
    assert filename.endswith('.npz')
    loader = np.load(filename)  # filename must end with .npz
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def get_raw_pixel_classes():
    #import nibabel as nib
    base_data_dir = "/home/jessica/Documents/hart-seg-ml/allrawnifti"
    example_segmentation = os.path.join(base_data_dir, 'trial8_30_fs_seg_ak5_lh4_TRANS.nii')
    scan_voxel = nib.load(example_segmentation)
    struct_arr = scan_voxel.get_data()
    n, h, w = struct_arr.shape
    class_labels = list(np.unique(struct_arr))
    
def check_one_hot(encoded_img):
    print(encoded_img.shape)
    return np.all(np.sum(encoded_img, axis=2) == 1.)

def batch_img_resize(images, h = 256, w = 256):
    images_resized = np.zeros([0, newHeight, newWidth], dtype=np.uint8)
    for  image in range(images.shape[0]):
        temp = imresize(images[image], [h, w], 'nearest')
        images_resized = np.append(images_resized, np.expand_dims(temp, axis=0), axis=0)
    return images_resized


# In[4]:


raw_pixel_classes =[0, 7, 8, 9, 45, 51, 52, 53, 68]  # Expected raw grayscale values for each pixel
#directory = "/Users/nozik/Documents/HARTresearch/allpreprocessed"
directory = "/home/jessica/Documents/hart-seg-ml/30_deg_training"
filenames = []  # Stores all filenames
raw_images = []  # Stores X (Raw cross section images as 2D np.ndarray)
segmentations = []  # Stores Y (Labeled/Segmented image as one-hot-encoded NumClasses-D np.ndarray)
h, w = 512, 512


# In[5]:


for folder in os.listdir(directory):
    class_labels = set()
    if not folder.startswith('.'):
        path = os.path.join(directory, folder)
        print(path)
        files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.startswith('.')])
        
        # Class label sanity check
#         for f in files:
#             if 'label' in f:
#                 img = imread(os.path.join(path, f), flatten=True)
#                 class_labels = class_labels.union(np.unique(img))
#         if not class_labels.issubset(raw_pixel_classes):
#             print("Class labels found in labeled images do not match the expected classes for scan {}".format(folder))
#             print("Expected {}".format(raw_pixel_classes))
#             print("Received {}".format(sorted(class_labels)))
#             break
        
        # Sanity image read and show some images in pairs (play with the range inputs)
#         for f in range(0, len(files), 2):
#             label_name = files[f]
#             raw_name = files[f+1]
#             label_img = imread(os.path.join(path, label_name), flatten=True)
#             raw_img = load_sparse_csr(os.path.join(path, raw_name)).toarray()  # Load sparse csr mat img -> to 2D numpy array
#             show_images([label_img, raw_img], titles=[label_name, raw_name])
        
        # Set up Datasets (X, Y) pairs of data ->
        # files are sorted by the name: either '#_label' or '#_raw'
        for f in files:
            print(f, end=' ')
            if 'label' in f:
                img = imread(os.path.join(path, f), flatten=True)
            else:
                img = load_sparse_csr(os.path.join(path, f)).toarray()            
            
#             imresize(seg[:,:,1],(h,w), interp='nearest')/255.0
            npad = ((15, 15), (58, 59))  # Pads to size 512, 512
            img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
            if 'raw' in f:
                raw_images.append(img)
            elif 'label' in f:
                encoded_img = one_hot_encode(img, raw_pixel_classes)
                segmentations.append(encoded_img)
            filenames.append(os.path.join(folder, f))
    print("")
            

# print(filenames)

            
# images = np.array(images)
# segmentations = np.round(np.array(segmentations)).astype('uint8')


# study_num = int(2)
# train_lst = np.load('data/splits/train_lst_' + str(study_num) + '.npy')
# val_lst = np.load('data/splits/val_lst_' + str(study_num) + '.npy')


# In[ ]:



# raw_90_deg_training = np.array(raw_images)
# seg_90_deg_training = np.array(segmentations)


# ### Split into Training, Cross Validation and Test sets

# In[5]:


print(len(raw_images), len(segmentations))


# In[6]:


"""
TODO: Same Scan cannot be used across Train, Validation and Test sets
TODO: Different weight conditions and angles may be used to segment other raw_scans
TODO: Bounding Box, image resizing, padding edges
"""
# raw_images holds our X data
# segmentations holds out Y data
x_train, y_train = [], []
x_val, y_val = [], []
x_test, y_test = [], []

percent_train, percent_val, percent_test = 60, 10, 30
num_train = np.round(len(raw_images) * percent_train/100).astype(np.int)
num_val = np.round(num_train + len(raw_images) * percent_val/100).astype(np.int)
num_test = np.round(num_val + len(raw_images) * percent_test/100).astype(np.int)

print("num_train: ", num_train, "num_val: ", num_val, "num_test: ", num_test)

assert len(raw_images) == len(segmentations)
rand_indices = list(np.random.choice(len(raw_images), len(raw_images), replace=False))

for i in rand_indices[:num_train]:
    x_train.append(raw_images[i])
    y_train.append(segmentations[i])
for j in rand_indices[num_train:num_val]:
    x_val.append(raw_images[j])
    y_val.append(segmentations[j])
for k in rand_indices[num_val:num_test]:
    x_test.append(raw_images[k])
    y_test.append(segmentations[k])

        
x_train = np.array(x_train).reshape((len(x_train), h, w, 1))
x_test = np.array(x_test).reshape((len(x_test), h, w, 1))
x_val = np.array(x_val).reshape((len(x_val), h, w, 1))
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)


print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

# Fix data padding to create square 482 by 482 matrix
# npad = ((0, 0), (15, 15), (58, 59), (0, 0))
# x_train = np.pad(x_train, pad_width=npad, mode='constant', constant_values=0)
# x_test = np.pad(x_test, pad_width=npad, mode='constant', constant_values=0)
# y_train = np.pad(y_train, pad_width=npad, mode='constant', constant_values=0)
# y_test = np.pad(y_test, pad_width=npad, mode='constant', constant_values=0)

# print()
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)


#     image = imresize(imread(directory + folder + '/' + folder + '.jpg', flatten = True),(h, w))
#     images.append(image)
#     filenames.append(folder)
#     seg = np.load(directory+folder+'/seg.npy')
#     temp = np.zeros((h,w,1))
#     temp[:,:,1] = imresize(seg[:,:,1],(h,w), interp='nearest')/255.0
#     segmentations.append(temp)


# # U-Net Model

# In[7]:



class Unet(object):        
    def __init__(self, mean, weight_decay, learning_rate, label_dim = 8, dropout = 0.9):
        self.x_train = tf.placeholder(tf.float32, [None, h, w, 1])
        self.y_train = tf.placeholder(tf.float32, [None, h, w, 9])
        self.x_test = tf.placeholder(tf.float32, [None, h, w, 1])
        self.y_test = tf.placeholder(tf.float32, [None, h, w, 9])
        
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.output = self.unet(self.x_train, mean, keep_prob=self.dropout)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.pred = self.unet(self.x_test, mean, reuse = True, keep_prob = 1.0)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
    
    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def unet(self, input, mean, keep_prob = 0.9, reuse = None):
        with tf.variable_scope('vgg', reuse=reuse):
            input = input - mean  # Demean
            
            pool_ = lambda x: nn.max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, padding = 'SAME', relu = True, filter_size = 3: nn.conv(x, filter_size, output_depth, 1, self.weight_decay, 
                                                                                                           name=name, padding=padding, relu=relu)
            deconv_ = lambda x, output_depth, name: nn.deconv(x, 2, output_depth, 2, self.weight_decay, name=name)
            
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

            conv_5_1 = conv_(pool_4, 1024, 'conv5_1')
            conv_5_2 = conv_(conv_5_1, 1024, 'conv5_2')
            
            pool_5 = pool_(conv_5_2)
            
            conv_6_1 = tf.nn.dropout(conv_(pool_5, 2048, 'conv6_1'), keep_prob)
            conv_6_2 = tf.nn.dropout(conv_(conv_6_1, 2048, 'conv6_2'), keep_prob)
            
            up_7 = tf.concat([deconv_(conv_6_2, 1024, 'up7'), conv_5_2], 3)  # Error here rn
            
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
            conv_11_2 = conv_(conv_11_1, 64, 'conv11_2')
            
            conv_12 = conv_(conv_11_2, 9, 'conv12_2', filter_size = 1, relu = False)
            return conv_12


# In[8]:


# Hyperparameters
mean = 0
weight_decay = 1e-6
learning_rate = 1e-4
label_dim = 8
maxout = False

# Create TF graph and initialize variables
tf.reset_default_graph()
sess = tf.Session()
model = Unet(mean, weight_decay, learning_rate, label_dim , dropout = 0.5)
sess.run(tf.global_variables_initializer())


# In[ ]:


# Restore old model
# saver = tf.train.Saver()
# saver.restore(sess, '/media/deoraid03/jeff/models/a4c_experiments/deep_256_2')


# In[9]:


# Train Model
nn.train(sess, model, x_train, y_train, x_val, y_val, epochs = 1000, batch_size = 1)


# In[12]:


import time


# In[14]:


start = time.time()
# IOU Accuracies for each label
print(nn.validate(sess, model, x_val, y_val))
end = time.time()
print(end - start)


# In[10]:


# Save model
models_dir = '/home/jessica/Documents/hart-seg-ml/models/'
model_name = '30_deg_training_sorted_inputs'
saver = tf.train.Saver()
saver.save(sess, os.path.join(os.path.join(models_dir, model_name), model_name))


# In[ ]:


# Load model
saver = tf.train.import_meta_graph('/home/jessica/Documents/hart-seg-ml/models/test_june18/test_june18.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/test_june18'))


# In[36]:


directory = "/home/jessica/Documents/hart-seg-ml/test"
filenames = []  # Stores all filenames
raw_images = []  # Stores X (Raw cross section images as 2D np.ndarray)
segmentations = []  # Stores Y (Labeled/Segmented image as one-hot-encoded NumClasses-D np.ndarray)

for folder in os.listdir(directory):
    if not folder.startswith('.'):
        path = os.path.join(directory, folder)
        files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.startswith('.')])
        
        for f in files:
            print(f, end=' ')
            if 'label' in f:
                img = imread(os.path.join(path, f), flatten=False)
            else:
                img = load_sparse_csr(os.path.join(path, f)).toarray()            
            
#             imresize(seg[:,:,1],(h,w), interp='nearest')/255.0
            npad = ((15, 15), (58, 59))  # Pads to size 512, 512
            img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
            if 'raw' in f:
                raw_images.append(img)
            elif 'label' in f:
                # encoded_img = one_hot_encode(img, raw_pixel_classes)
                # segmentations.append(encoded_img)
                segmentations.append(img)
            filenames.append(os.path.join(folder, f))


    


# In[37]:


raw_imgs_arr = np.array(raw_images)
raw_imgs_arr = np.expand_dims(raw_imgs_arr, axis=3)

seg_imgs_arr = np.array(segmentations)
#seg_imgs_arr = np.expand_dims(seg_imgs_arr, axis=3)



print(raw_imgs_arr.shape)

print(seg_imgs_arr.shape)


# In[41]:


def predict_cross_sec(x, model, sess):
    prediction = model.predict(sess, x)
    pred_classes = np.argmax(prediction[0], axis=2)
    return pred_classes

    
def predict_whole_seg(X, model, sess):
    '''
    Todo: Crop the predictions. 
    '''
    # This shape is hardcoded based on size of subject 1's scans.
    #segmented = np.empty((X.shape[0], 482, 395))
    segmented = np.empty(X.shape[:3])
    print("shape: ", segmented.shape)
    num_sections = X.shape[0]
    for i in range(num_sections):
        pred = predict_cross_sec(X[i:i+1], model, sess)
        # segmented[i] = crop_cross_sec(pred)
        segmented[i] = pred
        print(i, end=', ')
    return segmented


def crop_cross_sec(cross_sec):
    '''
    512, 512 -> 482, 395
    npad = ((15, 15), (58, 59))  # Pads to size 512, 512
    img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
    
    Hardcoded right now. Todo: remember orginal dimensions and parameterize.
    Currently all available scans have the same dimensions, so will work hardcoded for all current scans
    as of 06/13/18
    '''
    return cross_sec[15:512-15,58:512-59]
    

def convert_seg_to_nifti(seg):
    '''
    Hardcoded right now. Todo: generalize for any scan.
    '''
    base_data_dir = "/home/jessica/Documents/hart-seg-ml/allrawnifti"
    original_vol = nib.load(os.path.join(base_data_dir, 'trial15_60_w1_volume_TRANS.nii'))
    new_header = original_vol.header.copy()
    new_nifti = nib.nifti1.Nifti1Image(seg, None, header=new_header)
    save_dir = "/home/jessica/Documents/hart-seg-ml/predictedsegs/u-net_v1.0/30_deg_training"
    save_name = "trial15_60_w1_pred_seg.nii"
    nib.save(new_nifti, os.path.join(save_dir, save_name))
    
def convert_arr_to_nifti(arrs, orig_nii_dir, trial_name, save_dir, save_name, segmented=False):
    '''
    arr should be tuple of numpy arrays of shape (N, height, width). Vol first then seg.
    '''
    orig_nii_files = {}
    
    for file_name in os.listdir(orig_nifti_dir):
        if trial_name in file_name:
            if 'volume' in file_name:
                orig_nii_files['volume'] = file_name
            elif 'seg' in file_name:
                orig_nii_files['seg'] = file_name
    
    print(orig_nii_files)
    
    orig_vol_path = os.path.join(orig_nii_dir, orig_nii_files['volume'])
    orig_seg_path = os.path.join(orig_nii_dir, orig_nii_files['seg'])
    
    
    orig_vol_nii = nib.load(orig_vol_path)
    header = orig_vol_nii.header.copy()
    new_vol_nii = nib.nifti1.Nifti1Image(arrs[0], None, header=header)
    save_name = trial_name + "_proc_filled_volume.nii"
    nib.save(new_vol_nii, os.path.join(save_dir, save_name))
    
    orig_seg_nii = nib.load(orig_seg_path)
    header = orig_seg_nii.header.copy()
    new_seg_nii = nib.nifti1.Nifti1Image(arrs[1], None, header=header)
    save_name = trial_name + "_proc_filled_seg.nii"
    nib.save(new_seg_nii, os.path.join(save_dir, save_name))
    
#     for nii in orig_nii_files:
#         if 'volume' in nii and not segmented:
#             nii_path = os.path.join(orig_nifti_dir, nii)
#         elif 'seg' in nii and segmented:
#             nii_path = os.path.join(orig_nifti_dir, nii)
            
#     print(nii_path)
    
#     orig_nii = nib.load(nii_path)
#     header = orig_nii.header.copy()
#     new_nii = nib.nifti1.Nifti1Image(arr, None, header=header)
    
#     nib.save(new_nii, os.path.join(save_dir, save_name))


# In[39]:


print(raw_imgs_arr.shape)
all_segs = predict_whole_seg(raw_imgs_arr, model, sess)


# In[40]:


print(all_segs.shape)


# In[42]:


convert_seg_to_nifti(all_segs)


# In[37]:


new_shape = raw_imgs_arr.shape[:3]
convert_seg_to_nifti(raw_imgs_arr.reshape(new_shape))


# In[28]:


data_arrs = (raw_imgs_arr, seg_imgs_arr)

orig_nifti_dir = "/home/jessica/Documents/hart-seg-ml/allrawnifti"
curr_trial = "trial20_90_w1"
save_dir = "/home/jessica/Documents/hart-seg-ml/allrawfillednifti"
# save_name = "trial8_30_fs_proc_filled_vol.nii"
save_name = None
convert_arr_to_nifti(data_arrs, orig_nifti_dir, curr_trial, save_dir, save_name)

