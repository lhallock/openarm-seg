# Ultrasound Muscle Segmentation

This repo contains the code used in the development of the OpenArm 2.0 data set, a set of volumetric scans of the human arm. Contained in the repo are various scripts that can be used to train a U-Net model on volumetric ultrasound data, generate predictions on a large dataset, and collect information about the quality of the predictions.

If you use this code for academic purposes, please cite the following publication: Yonatan Nozik\*, Laura A. Hallock\*, Daniel Ho, Sai Mandava, Chris Mitchell, Thomas Hui Li, and Ruzena Bajcsy, "OpenArm 2.0: Automated Segmentation of 3D Tissue Structures for Multi-Subject Study of Muscle Deformation Dynamics," in International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), IEEE, 2019. \*equal contribution

## Installation

The following Python modules are required to run this code: `numpy`, `tensorflow`, `os`, `sys`, `math`, `logging`, `argparse`, `configparser`, `pickle`, `shutil`, `nibabel`, `scipy`, `gc`, `time`, `timeit`, `collections`.



## Setup

Most of the scripts provided here assume that you have source data in the following directory structure somewhere on your machine:

```bash
├── models
│   ├── u-net_v1-0
│   │   ├── model_1
│   │   ├── model_2
│   │   └── folder_holding_multiple_models
│   │       ├── submodel_1
│   │       ├── submodel_2
│   │       └── ...
│	├── ...
├── SubA
│   ├── all_nifti
│   ├── predictions
│   │   └── under_512
│   │       ├── group_4_5
│   │       └── ...
│   ├── prediction_sources
│   │   ├── over_512
│   │   └── under_512
│   │       ├── trial10_30_p5
│   │       ├── trial1_0_fs
│   │       └── ...
├── SubB
│   ├── all_nifti
│   ├── predictions
│   │   ├── over_512
│   │   │   ├── group_10_1_17extra
│   │   │   ├── group_1_1
│   │   │   └── ...
│   │   └── under_512
│   │       ├── group_10_1_17extra
│   │       ├── group_1_1
│   │       └── ...
│   ├── prediction_sources
│   │   ├── over_512
│   │   │   ├── trial11_60_fs
│   │   │   ├── trial12_60_gc
│   │   │   └── ...
│   │   ├── under_512
│   │   │   ├── trial10_30_p5
│   │   │   ├── trial1_0_fs
│   │   │   └── ...
├── Rest of the Sub[x]s...
└── training_groups
    ├── model_1
    │   ├── model_1_1
    │   │   ├── trial11_60_fs
    │   │   └── ...
    │   ├── model_1_2
    │   │   ├── trial11_60_fs
    │   │   └── ...
    │   └── model_1_3
    │       ├── trial1_0_fs
    │       └── ...
    ├── model_2
    │   ├── model_2_1
    │   │   └── trial6_30_fs
    │   ├── model_2_2
    │   │   ├── trial6_30_fs
    │   │   └── trial7_30_gc
    │   └── model_2_3
    │       ├── trial10_30_p5
    │       ├── trial10_30_p5_ed
    │       ├── trial9_30_p3
    │       └── ...
    ├── Rest of the groups...
    └── trial_names
        ├── trial1_0_fs
        ├── trial11_60_fs
        ├── trial16_90_fs
        ├── trial2_0_gc
        ├── trial3_0_p1
        ├── trial4_0_p3
        ├── trial5_0_p5
        ├── trial6_30_fs
        ├── trial7_30_gc
        ├── trial8_30_p1
        └── trial9_30_p3

```

The provided scripts which automate tasks such as training multiple models, generating segmentations of multiple images over multiple models, and scraping data from trained models or generated segmentations depend on this structure. 

### Models

The `models` directory is where the Tensorflow model files are stored. Each subdirectory such as `u-net_v1-0` holds all trained models of the same architecture. Inside `u-net_v1-0` are the folders holding the actual Tensorflow model files, so for example `model_1` would hold `checkpoint`, `model_1.data`, `model_1.index`, `model_1.meta`, as well as some other information about the model that gets stored such as plaintext and pickled files with loss per epoch, validation accuracies, etc. 

### Subjects

The dataset used for this project involved 20 ultrasound scans from 11 participants, labeled subjects A through k (SubA through SubK). Each subject has their own folder with the same minimum structure. For example, in the SubA directory, `all_nifti` contains all the source NiFTI files corresponding to that subject. This will be a minimum of 20 images, however it may be up to 40 depending on if some of the images were manually segmented by humans. The raw images and manually segmented images can be distinguished by the `_volume` and `_seg` suffixes, respectively. 

### Training Sources

The `training_groups` subdirectory structure shown above is only partially complete. Below is the internal structure of the `model_1` directory (`model_2` is analogous):

```bash
├── model_1_1
│   ├── trial1_0_fs
│   │   ├── trial1_0_fs_seg.nii
│   │   └── trial1_0_fs_volume.nii
│   ├── trial11_60_fs
│   │   ├── trial11_60_fs_seg_cropped.nii
│   │   └── trial11_60_fs_volume_cropped.nii
│   ├── trial16_90_fs
│   │   ├── trial16_90_fs_seg_cropped.nii
│   │   └── trial16_90_fs_volume_cropped.nii
│   └── trial6_30_fs
│       ├── trial6_30_fs_seg.nii
│       └── trial6_30_fs_volume.nii
├── model_1_2
│   ├── trial1_0_fs
│   │   ├── trial1_0_fs_seg.nii
│   │   └── trial1_0_fs_volume.nii
│   ├── trial1_0_fs_ed
│   │   ├── trial1_0_fs_seg_ed.nii
│   │   └── trial1_0_fs_vol_ed.nii
│   ├── trial11_60_fs
│   │   ├── trial11_60_fs_seg_cropped.nii
│   │   └── trial11_60_fs_volume_cropped.nii
│   ├── trial11_60_fs_ed
│   │   ├── trial11_60_fs_seg_ed.nii
│   │   └── trial11_60_fs_vol_ed.nii
│   ├── trial16_90_fs
│   │   ├── trial16_90_fs_seg_cropped.nii
│   │   └── trial16_90_fs_volume_cropped.nii
│   ├── trial16_90_fs_ed
│   │   ├── trial16_90_fs_seg_ed.nii
│   │   └── trial16_90_fs_vol_ed.nii
│   ├── trial6_30_fs
│   │   ├── trial6_30_fs_seg.nii
│   │   └── trial6_30_fs_volume.nii
│   └── trial6_30_fs_ed
│       ├── trial6_30_fs_seg_ed.nii
│       └── trial6_30_fs_volume_ed.nii
```

When using `training.py` to train a new model, you need to give the path to a directory with the structure of `model_1_1` or `model_1_2`. The name of this folder and its `trial` subfolders are labeled only for convenience. The important part is that each `model_x_y` directory holds all of the training data that will be used to train one particular model. Each `trial` subfolder holds both the raw image data as well as the segmented ground truth corresponding to that raw image.

## Model Training 

### Training a single model

To train a model, first step up a directory containing the segmentations you wish to include for training. See the previous section for details on what structure is expected by the training scripts.

You can change the parameters for the training session through a config file `trainingconfig.ini`. If this file does not already exist then you can run the script once to generate it. Two important parameters that are not yet controllable via the config file are the total number of 2d image slices that will be kept by the script for training. These need to be modified in the source code of `training.py` directly by changing the `total_keep` arguments passed into the two calls to `split_data`. Once you have set up the directory containing your data for training and you have the configuration file, create a section in the configuration file for your model. `models_dir` is the directory where the new model folder will be created. `training_data_dir` is the path to the directory where you have set up your desired training data (in the example above this would be the path to `model_1_1`, `model_2_1`, etc). To begin training, call the script from the command line with: `python training.py name_of_your_model -s section_name_in_trainingconfig`. The model name and name of the section in `trainingconfig.ini` do not need to be the same.

### Queuing training for multiple models

To set up training for multiple models at once (that is, train any number of models consecutively), follow the steps above to create the training data directory and training config info for the desired models. Then modify the bash script `trainmultiple.sh`, which is just putting each command for an individual model one after the other.

## Predicting Segmentations

Models created with the provided `training.py` script are saved using `tf.train.Saver` and so can be restored with `tf.train.Saver.restore()`. The provided `save_model` and `load_model` methods in `pipeline.py` can be used to work with a single or small number of models.

Once a model is loaded, `predict_whole_seg` can be used to generate a prediction of a single NIfTI scan. Alternatively, `predict_all_segs` can be used to generate segmentations for all NIfTIs in a given directory.

The `predict_all_groups.py` script is the easiest way to generate multiple segmentations for multiple models. The directory structure detailed above must be in place in order for this script to function properly, in particular the structure of the directories `SubA`, `SubB`, etc. In order to change which segmentations to generate, modify the `over_512_configs` and `over_512_configs` lists. Each element in these lists is a 3-tuple in which the first element is the path to where the ground truth data to be used for prediction is stored, the second element is the path to where the predictions will be saved, and the third element is the path to all the available ground truth files for that particular subject. There is a separation of "over 512" and "under 512" where applicable for each subject since some scans, in order to be padded to a power of two, must be padded up to 1024 while others only require to be padded up to 512. These can't coexist in the code as it currently exists and so the separation is a workaround of this issue. 

The last two modifications of this script that will be necessary in order to run on your machine will be to change the assignment of `models_dir` to one for your machine. Place all model folders you want to use for prediction in this directory. As shown in the directory structure above, this could be the path to `folder_holding_multiple_models` or `u-net_v1-0`, for example. It will not search recursively, so in the latter case no models in `folder_holding_multiple_models` will be included. Lastly, you will need to change the variable `group_whitelist` to include the names of the models (the name of the folders holding the model files) to the ones you want to be used for prediction. In the directory structure shown above, if you provided the path to `folder_holding_multiple_models`, you would add `submodel_1` if you want `submodel_1` to be included and omit `submodel_2` to exlude it from prediction. This whitelist was convenient for the development process but you can remove it by removing the check: `if group not in group_whitelist`. 