# Ultrasound Muscle Segmentation via Convolutional Neural Networks

This repo contains the code used in the development of the OpenArm 2.0 data set, a set of volumetric scans of the human arm. Contained in the repo are various scripts that can be used to train a U-Net model on volumetric ultrasound data, generate predictions on a large dataset, and collect information about the quality of the predictions. The code is designed to permit batch training of different network architectures and training data sets to best predict individual subjects of interest.

If you use this code for academic purposes, please cite the following publication: Yonatan Nozik\*, Laura A. Hallock\*, Daniel Ho, Sai Mandava, Chris Mitchell, Thomas Hui Li, and Ruzena Bajcsy, "OpenArm 2.0: Automated Segmentation of 3D Tissue Structures for Multi-Subject Study of Muscle Deformation Dynamics," in _International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)_, IEEE, 2019. \*Equal contribution.

This README primarily describes the usage of our CNN-based segmentation methods that are the focus of the publication above. In addition, the code used to perform baseline image registration is included, and its usage is documented in the _Registration-Based Segmentation_ section at the bottom of this page.

The documentation below is intended foremost as a record of the code used for OpenArm 2.0 generation and is provided as-is. However, we invite anyone who wishes to adapt and use the code below under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Installation

The following Python modules are required to run this code: `numpy`, `tensorflow`, `os`, `sys`, `math`, `logging`, `argparse`, `configparser`, `pickle`, `shutil`, `nibabel`, `scipy`, `gc`, `time`, `timeit`, and `collections`.



## Setup

Source data and models should be organized in the following directory structure:

```bash
├── models
│   ├── u-net_v1-0
│   │   ├── model_1
│   │   ├── model_2
│   │   └── folder_holding_multiple_models
│   │       ├── model_3
│   │       ├── model_4
│   │       └── ...
│	├── ...
├── Sub1
│   ├── all_nifti
│   ├── predictions
│   │   ├── over_512
│   │   │   ├── group_1_1
│   │   │   ├── group_1_2
│   │   │   └── ...
│   │   └── under_512
│   │       ├── group_1_1
│   │       ├── group_1_2
│   │       └── ...
│   ├── prediction_sources
│   │   ├── over_512
│   │   │   ├── trial11_60_fs
│   │   │   ├── trial12_60_gc
│   │   │   └── ...
│   │   ├── under_512
│   │   │   ├── trial1_0_fs
│   │   │   ├── trial10_30_p5
│   │   │   └── ...
├── Additional Sub[x] folders...
└── training_groups
    ├── group_1
    │   ├── group_1_1
    │   │   ├── trial11_60_fs
    │   │   └── ...
    │   ├── group_1_2
    │   │   ├── trial11_60_fs
    │   │   └── ...
    │   └── group_1_3
    │       ├── trial1_0_fs
    │       └── ...
    ├── group_2
    │   ├── group_2_1
    │   │   └── trial6_30_fs
    │   ├── group_2_2
    │   │   ├── trial6_30_fs
    │   │   └── trial7_30_gc
    │   └── group_2_3
    │       ├── trial10_30_p5
    │       ├── trial10_30_p5_ed
    │       ├── trial9_30_p3
    │       └── ...
    └── Additional training groups...

```

(The example structure above is reasonably comprehensive, aside from the
`training_groups` folder, which is fleshed out more comprehensively below.)

Broadly, these directories contain the models used for training (`models`), 3D data and predictions for each target subject (`Sub1`, `Sub2`,...), and data used for training each specified network configuration (`training_groups`).

While some simpler training functionality can be executed without this structure --- and some aspects of this structure are more fungible than others --- it is necessary for scripts that automate training multiple models, generating segmentations of multiple images over multiple models, and scraping data from trained models or generated segmentations.

Each directory is described in further detail below.

### Models

TensorFlow model files are stored in the `models` directory. Each subdirectory (e.g., `u-net_v1-0`) should contain all trained models of the same architecture. Inside each subdirectory are folders holding the actual Tensorflow model files, including both specified models and data that is logged during training. For example, `model_1` holds `checkpoint`, `model_1.data`, `model_1.index`, and `model_1.meta`, as well as the other information about the model that is stored during training (i.e., plain text and pickled files with loss per epoch, validation accuracies, etc.). Models can also be organized in a deeper folder structure if desired, provided the directory is properly specified in `trainingconfig.ini`, as described in the _Model Training_ section below.

Our own pre-trained models used in the publication above are available for download as part of the [OpenArm repository](https://simtk.org/frs/?group_id=1617). Other models may be used as well.

### Subjects

Subject folders `Sub[x]` contain all volumetric data associated with a given subject, including both raw volumetric scans and those that are populated by the network at prediction time. Subfolder `all_nifti` contains all raw NIfTI scans for which a predicted segmentation is desired (`*_volume.nii`), as well as (if available) ground-truth segmentations for each volume (`*_seg.nii`). The latter can be used to assess prediction quality, as described in _Assessing Segmentation Quality_ below.

Each subject's `predictions` folder is populated at prediction time, though the
directory itself (and its `over_512` and `under_512` subdirectories) must be
created in advance. Specifically, when predictions are executed for a
particular "group" (for which models and training data sources are
specified in `trainingconfig.ini`), these prediction files are written
into corresponding subfolders within each `predictions` folder.

Note that the `over_512` and `under_512` directories separate scans of which predicted slices are larger and smaller than 512x512 pixels. This is an artifact of the way the neural network generates predictions: scans smaller than 512x512 are passed directly through the network, while those larger are first cropped, and must thus be resized when reassembled into the final predicted NiFTI. The full pipeline places them into separate folders, though this can easily be modified if desired.

Each subject's `prediction_sources` folder contains... (TODO)

To predict segmentations for the available OpenArm 2.0 scans, first download all desired subject archives from the [project website](https://simtk.org/frs/?group_id=1617). All volume files for which predictions are desired (`Sub[x]/volumes/*_volume.mha`) should then be converted to the NIfTI file format (e.g., using [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php), renamed to follow convention `trial[n]_*_volume.nii`, and placed in the `all_nifti` folder. Available ground truth scans (`Sub[x]/ground_segs/*.nii` may also be placed in this folder if available. 

### Training Sources

The `training_groups` directory contains all data used to train each "group" (i.e. specified network and training data set for analysis). Below is an example subdirectory structure (for `training_groups/group_1` above; `group_2` is analogous):

```bash
group_1
├── group_1_1
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
├── group_1_2
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

Each `group_[j]_[k]` folder contains all data used for training that group, structured as a series of subfolders, each containing a single 3D volumetric scan and its associated segmentation. In general, structure below each `group_[j]_[k]` subfolder is not important, only that all volumes and their corresponding segmentations are included. TODO: Are the trial prefixes important? Are the subfolders with a single volume and segmentation important? How do you tell the difference between vol and seg? Suffix?

## Model Training 

### Training a Single Model

To train a model, first step up a directory containing the segmentations you wish to include for training. See the previous section for details on what structure is expected by the training scripts.

You can change the parameters for the training session through a config file `trainingconfig.ini`. If this file does not already exist then you can run the script once to generate it. Two important parameters that are not yet controllable via the config file are the total number of 2d image slices that will be kept by the script for training. These need to be modified in the source code of `training.py` directly by changing the `total_keep` arguments passed into the two calls to `split_data`. Once you have set up the directory containing your data for training and you have the configuration file, create a section in the configuration file for your model. `models_dir` is the directory where the new model folder will be created. `training_data_dir` is the path to the directory where you have set up your desired training data (in the example above this would be the path to `model_1_1`, `model_2_1`, etc). To begin training, call the script from the command line with: `python training.py name_of_your_model -s section_name_in_trainingconfig`. The model name and name of the section in `trainingconfig.ini` do not need to be the same.

### Queuing Training for Multiple Models

To set up training for multiple models at once (that is, train any number of models consecutively), follow the steps above to create the training data directory and training config info for the desired models. Then modify the bash script `trainmultiple.sh`, which is just putting each command for an individual model one after the other.

## Predicting Segmentations

Models created with the provided `training.py` script are saved using `tf.train.Saver` and so can be restored with `tf.train.Saver.restore()`. The provided `save_model` and `load_model` methods in `pipeline.py` can be used to work with a single or small number of models.

Once a model is loaded, `predict_whole_seg` can be used to generate a prediction of a single NIfTI scan. Alternatively, `predict_all_segs` can be used to generate segmentations for all NIfTIs in a given directory.

The `predict_all_groups.py` script is the easiest way to generate multiple segmentations for multiple models. The directory structure detailed above must be in place in order for this script to function properly, in particular the structure of the directories `SubA`, `SubB`, etc. In order to change which segmentations to generate, modify the `over_512_configs` and `over_512_configs` lists. Each element in these lists is a 3-tuple in which the first element is the path to where the ground truth data to be used for prediction is stored, the second element is the path to where the predictions will be saved, and the third element is the path to all the available ground truth files for that particular subject. There is a separation of "over 512" and "under 512" where applicable for each subject since some scans, in order to be padded to a power of two, must be padded up to 1024 while others only require to be padded up to 512. These can't coexist in the code as it currently exists and so the separation is a workaround of this issue. 

The last two modifications of this script that will be necessary in order to run on your machine will be to change the assignment of `models_dir` to one for your machine. Place all model folders you want to use for prediction in this directory. As shown in the directory structure above, this could be the path to `folder_holding_multiple_models` or `u-net_v1-0`, for example. It will not search recursively, so in the latter case no models in `folder_holding_multiple_models` will be included. Lastly, you will need to change the variable `group_whitelist` to include the names of the models (the name of the folders holding the model files) to the ones you want to be used for prediction. In the directory structure shown above, if you provided the path to `folder_holding_multiple_models`, you would add `submodel_1` if you want `submodel_1` to be included and omit `submodel_2` to exlude it from prediction. This whitelist was convenient for the development process but you can remove it by removing the check: `if group not in group_whitelist`. 

## Assessing Segmentation Quality

TODO

## Training with Augmented Data

TODO

## Registration-Based Segmentation

In addition to the CNN-based segmentation code above, we provide the registration-based segmentation code, built using [SimpleElastix](https://simpleelastix.github.io/), used as a baseline in the publication above. Its use is documented below.

Note that registration-based segmentation consists of mapping the segmented tissue structures of one scan to another by finding the optimal transformation between the the two raw images. An excellent description of this process can be found [here](https://simpleelastix.readthedocs.io/Introduction.html).

### Installation

Use of registration code relies on the `numpy` Python module and the [SimpleElastix](https://simpleelastix.github.io/) library.

### Setup

Edit the following parameters, at the top of the `run_amsaf` method in `registration.py`:

`verbose` -- set as `True` or `False` based on desired verbosity

`unsegmented_image` -- set as `read_image("[image_to_be_segmented].nii")` based on the file path of the NIfTI to be segmented

`segmented_image`, `segmentation` -- set as `read_image("[segmented_image].nii")` based on the file paths of the segmented NIfTI source image and its associated segmentation, respectively

`new_segmentation` -- set as desired output file name for new segmentation

If your segmented and unsegmented images are not already roughly aligned, you may choose to specify a manual affine transformation with which to initialize the registration process by modifying the `A` and `t` parameters.

By default, the provided code will perform a hierarchy of rigid, affine, and nonlinear transformations, with the result of each registration initializing the next. If you wish to more precisely control the behavior of these transformations, you may edit the `DEFAULT_*` parameter maps included at the bottom of `registration.py`.

### Usage

Run

```bash
python registration.py
```
