import random

def load_data(directory, views, view_index):
    '''
    Load images from directory and store labels for each image in python lists
    
    @params directory: String of absolute file path
    @params views: List of views that we want to load
    @params view_index: List of class labels for each view in 'views'
    '''
    images = []
    labels = []
    filenames = []

    for filename in os.listdir(directory):
        for i in range(len(views)):
            if views[i] in filename:
                images.append(imread(directory+filename, flatten = True).astype('uint8'))
                filenames.append(filename)
                # Create a one-hot encoding vector for labels
                label = [0] * len(views)
                label[view_index[i]] = 1
                labels.append(label)

    return images, labels, filenames
    
def split_data(filenames, images, labels, train_lst = None, val_lst = None, train_perc = 0.8):
    '''
    Split data into training and validation sets.
    If train_lst and val_lst are provided, sets will be split based on those studies.
    Else, studies will be split into the training set with 'train_perc' chance.
    
    @params filenames: List of filenames to be split
    @params images: List of images as numpy arrays
    @params train_lst, val_lst: List of strings denoting the studies in the train and validation sets
    @params train_perc: Float denoting percentage that a new study will be used as training
    '''
    # Store which studies we have seen before
    new = False
    if train_lst is None:
        new = True
        train_lst, val_lst = [], []
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(len(filenames)):
        filename = filenames[i]
        study = '_'.join(filename.split('_')[:4])
        #if 'Image' in filename:
         #   study = filename.split('Image')[0]
        #if 'image' in filename:
         #   study = filename.split('image')[0]
        if study in train_lst:
            x_train.append(images[i])
            y_train.append(labels[i])
        elif study in val_lst:
            x_test.append(images[i])
            y_test.append(labels[i])
        elif new:
            # We set new study as training data with probability 'train_perc'
            if random.random() <= train_perc:
                train_lst.append(study)
                x_train.append(images[i])
                y_train.append(labels[i])
            else:
                val_lst.append(study)
                x_test.append(images[i])
                y_test.append(labels[i])
    return x_train, y_train, x_test, y_test

def split_filenames(filenames, train_lst = None, val_lst = None, train_perc = 0.8):
    '''
    Split data into training and validation sets.
    If train_lst and val_lst are provided, sets will be split based on those studies.
    Else, studies will be split into the training set with 'train_perc' chance.
    
    @params filenames: List of filenames to be split
    @params images: List of images as numpy arrays
    @params train_lst, val_lst: List of strings denoting the studies in the train and validation sets
    @params train_perc: Float denoting percentage that a new study will be used as training
    '''
    # Store which studies we have seen before
    new = False
    if train_lst is None:
        new = True
        train_lst, val_lst = [], []
    filename_train = []
    filename_test = []
    for i in range(len(filenames)):
        filename = filenames[i]
        study = '_'.join(filenames[i].split('_')[:4])
        if study in train_lst:
            filename_train.append(filenames[i])
        else:
            filename_test.append(filenames[i])
    return filename_train, filename_test