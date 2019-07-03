import SimpleITK as sitk
import numpy as np
import os, sys, time



def run_amsaf():
    '''
    EDIT THESE VARIABLES
    '''
    verbose = True
    unsegmented_image = read_image("")
    segmented_image = read_image("")
    segmentation = read_image("")
    new_segmentation = "test_seg.nii"



    # Affine manual pre-registration/initialization such that x' = Ax + t
    A = np.array([[1.0, 0, 0],
                    [0, 1.0, 0], 
                    [0, 0.0, 1.0]])
    t = np.array([[0, 0, 0]])
    


    '''
    DO NOT EDIT BELOW HERE
    '''


    segmentation = transform(segmentation, init_affine_transform(segmentation, A, t), verbose)
    segmented_image = transform(segmented_image, init_affine_transform(segmented_image, A, t), verbose)


    result = segment(unsegmented_image, segmented_image, segmentation, get_default_parameter_maps(), verbose=verbose)
    write_image(result, new_segmentation)


def init_affine_transform(img, A, t):
    """Initializes an affine transform parameter map for a given image.

    The transform fits the following format: T(x) = A(x-c) + c + t
    This code uses c == 0 as a one could always find an equivalent translation

    :param img: Image to be transformed
    :param A: 3x3 numpy array consisting of a rotation matrix
    :param t: 1x3 numpy array consisting of the translational values
    :returns: SimpleITK.ParameterMap
    :type img: SimpleITK.Image
    :type A: numpy.ndarray
    :type t: numpy.ndarray
    :rtype: dict
    """
    affine = get_default_affine_transform()

    f = lambda x: tuple([str(i) for i in x])
    affine['Size'] = f(img.GetSize())
    affine['Spacing'] = f(img.GetSpacing())

    affine['Origin'] = f(img.GetOrigin())
    affine['Direction'] = f(img.GetDirection())

    affine['CenterOfRotationPoint'] = f(np.array([0,0,0]))

    transform = np.concatenate((A, t), axis=0)

    affine['TransformParameters'] = f(transform.ravel())
    return affine


def register(fixed_image,
             moving_image,
             parameter_maps,
             auto_init=True,
             verbose=False):
    """Register images using Elastix.

    :param parameter_maps: Vector of 3 parameter maps to be used for
                           registration. If none are provided, a default vector
                           of [rigid, affine, bspline] parameter maps is used.
    :param auto_init: Auto-initialize images. This helps with flexibility when
                      using images with little overlap.
    :param verbose: Flag to toggle stdout printing from Elastix
    :type fixed_image: SimpleITK.Image
    :type moving_image: SimpleITK.Image
    :type parameter_maps: [SimpleITK.ParameterMap]
    :type auto_init: bool
    :type verbose: bool
    :returns: Tuple of (result_image, transform_parameter_maps)
    :rtype: (SimpleITK.Image, [SimpleITK.ParameterMap])
    """
    registration_filter = sitk.ElastixImageFilter()
    if not verbose:
        registration_filter.LogToConsoleOff()
    registration_filter.SetFixedImage(fixed_image)
    registration_filter.SetMovingImage(moving_image)

    if auto_init:
        parameter_maps = _auto_init_assoc(parameter_maps)
    registration_filter.SetParameterMap(parameter_maps)
    for m in parameter_maps[1:]:
        registration_filter.AddParameterMap(m)

    registration_filter.Execute()
    result_image = registration_filter.GetResultImage()
    transform_parameter_maps = registration_filter.GetTransformParameterMap()

    return result_image, transform_parameter_maps


def segment(unsegmented_image,
            segmented_image,
            segmentation,
            parameter_maps,
            verbose=False):
    """Segment image using Elastix

    :param segmented_image: Image with corresponding segmentation passed as
                            the next argument
    :param segmentation: Segmentation to be mapped from segmented_image to
                         unsegmented_image
    :param parameter_maps: Optional vector of 3 parameter maps to be used for
                           registration. If none are provided, a default vector
                           of [rigid, affine, bspline] parameter maps is used.
    :param verbose: Flag to toggle stdout printing from Elastix
    :type unsegmented_image: SimpleITK.Image
    :type segmented_image: SimpleITK.Image
    :type segmentation: SimpleITK.Image
    :type parameter_maps: [SimpleITK.ParameterMap]
    :type verbose: bool
    :returns: Segmentation mapped from segmented_image to unsegmented_image
    :rtype: SimpleITK.Image
    """
    _, transform_parameter_maps = register(
        unsegmented_image, segmented_image, parameter_maps, verbose=verbose)

    return transform(
        segmentation, _nn_assoc(transform_parameter_maps), verbose=verbose)



def transform(image, parameter_maps, verbose=False):
    """Transform an image according to some vector of parameter maps

    :param image: Image to be transformed
    :param parameter_maps: Vector of 3 parameter maps used to dictate the
                           image transformation
    :type image: SimpleITK.Image
    :type parameter_maps: [SimpleITK.ParameterMap]
    :returns: Transformed image
    :rtype: SimpleITK.Image
    """
    transform_filter = sitk.TransformixImageFilter()
    if not verbose:
        transform_filter.LogToConsoleOff()
    transform_filter.SetTransformParameterMap(parameter_maps)
    transform_filter.SetMovingImage(image)
    transform_filter.Execute()
    image = transform_filter.GetResultImage()
    return image


def read_image(path, ultrasound=True):
    """Load image from filepath as SimpleITK.Image

    :param path: Path to .nii file containing image.
    :param ultrasound: Optional. If True, image will be cast as sitkUInt16 for ultrasound images.
    :type path: str
    :type ultrasound: bool
    :returns: Image object from path
    :rtype: SimpleITK.Image
    """
    image = sitk.ReadImage(path)
    if ultrasound:
        image = sitk.Cast(image, sitk.sitkUInt16)
    return image


def write_image(image, path):
    """Write an image to file

    :param image: Image to be written
    :param path: Destination where image will be written to
    :type image: SimpleITK.Image
    :type path: str
    :rtype: None
    """
    sitk.WriteImage(image, path)


##########################
# Private module helpers #
##########################



def _to_elastix(pm, ttype):
    elastix_pm = sitk.GetDefaultParameterMap(ttype)
    if sys.version_info[0] >=3:
        it = pm.items()
    else:
        it = pm.iteritems()
    for k, v in it:
        if type(v) == list:
            elastix_pm[k] = v
        else:
            elastix_pm[k] = [v]
    return elastix_pm



def _nn_assoc(pms):
    return _pm_vec_assoc('ResampleInterpolator',
                         'FinalNearestNeighborInterpolator', pms)

def _auto_init_assoc(pms):
    return _pm_vec_assoc('AutomaticTransformInitialization', 'true', pms)

def _pm_assoc(k, v, pm):
    result = {}
    if sys.version_info[0] >=3:
        it = pm.items()
    else:
        it = pm.iteritems()
    for key, val in it:
        if key == k:
            result[key] = [v]
        else:
            result[key] = val
    return result


def _pm_vec_assoc(k, v, pms):
    return [_pm_assoc(k, v, pm) for pm in pms]



def get_default_parameter_maps():
    return [DEFAULT_RIGID, DEFAULT_AFFINE, DEFAULT_BSPLINE]


def get_default_affine_transform():
    return DEFAULT_AFFINE_TRANSFORM

DEFAULT_RIGID = {
    "AutomaticParameterEstimation": ['true'],
    "AutomaticTransformInitialization": ['true'],
    "BSplineInterpolationOrder": ['3.000000'],
    "CheckNumberOfSamples": ['true'],
    "DefaultPixelValue": ['0.000000'],
    "FinalBSplineInterpolationOrder": ['3.000000'],
    "FixedImagePyramid": ['FixedSmoothingImagePyramid'],
    "ImageSampler": ['RandomCoordinate'],
    "Interpolator": ['BSplineInterpolator'],
    "MaximumNumberOfIterations": ['1024.000000'],
    "MaximumNumberOfSamplingAttempts": ['8.000000'],
    "Metric": ['AdvancedMattesMutualInformation'],
    "MovingImagePyramid": ['MovingSmoothingImagePyramid'],
    "NewSamplesEveryIteration": ['true'],
    "NumberOfHistogramBins": ['64.000000'],
    "NumberOfResolutions": ['3.000000'],
    "NumberOfSamplesForExactGradient": ['4096.000000'],
    "NumberOfSpatialSamples": ['2000.000000'],
    "Optimizer": ['AdaptiveStochasticGradientDescent'],
    "Registration": ['MultiResolutionRegistration'],
    "ResampleInterpolator": ['FinalBSplineInterpolator'],
    "Resampler": ['DefaultResampler'],
    "ResultImageFormat": ['nii'],
    "RequiredRatioOfValidSamples": ['0.05'], 
    #"Scales": ['Float'],
    "Transform": ['EulerTransform'],
    "WriteIterationInfo": ['false'],
    "WriteResultImage": ['true'],
  }

DEFAULT_AFFINE = {
    "AutomaticParameterEstimation": ['true'],
    "AutomaticScalesEstimation": ['true'],
    "CheckNumberOfSamples": ['true'],
    "DefaultPixelValue": ['0.000000'],
    "FinalBSplineInterpolationOrder": ['3.000000'],
    "FixedImagePyramid":
        ['FixedSmoothingImagePyramid'],
    "ImageSampler": ['RandomCoordinate'],
    "Interpolator": ['BSplineInterpolator'],
    "MaximumNumberOfIterations": ['1024.000000'],
    "MaximumNumberOfSamplingAttempts": ['8.000000'],
    "Metric": ['AdvancedMattesMutualInformation'],
    "MovingImagePyramid": ['MovingSmoothingImagePyramid'],
    "NewSamplesEveryIteration": ['true'],
    "NumberOfHistogramBins": ['32.000000'],
    "NumberOfResolutions": ['4.000000'],
    "NumberOfSamplesForExactGradient": ['4096.000000'],
    "NumberOfSpatialSamples": ['2048.000000'],
    "Optimizer": ['AdaptiveStochasticGradientDescent'],
    "Registration": ['MultiResolutionRegistration'],
    "ResampleInterpolator": ['FinalBSplineInterpolator'],
    "Resampler": ['DefaultResampler'],
    "ResultImageFormat": ['nii'],
    "RequiredRatioOfValidSamples": ['0.05'],
    "Transform": ['AffineTransform'],
    "WriteIterationInfo": ['false'],
    "WriteResultImage": ['true'],
  }

DEFAULT_BSPLINE = {
    'AutomaticParameterEstimation': ["true"],
    'CheckNumberOfSamples': ["true"],
    'DefaultPixelValue': ['0.000000'],
    'FinalBSplineInterpolationOrder': ['3.000000'],
    'FinalGridSpacingInPhysicalUnits': ['4.000000'],
    'FixedImagePyramid': ['FixedSmoothingImagePyramid'],
    'GridSpaceSchedule': ['2.803220 1.988100 1.410000 1.000000'],
    'ImageSampler': ['RandomCoordinate'],
    'Interpolator': ['LinearInterpolator'],
    'MaximumNumberOfIterations': ['1024.000000'],
    'MaximumNumberOfSamplingAttempts': ['8.000000'],
    'Metric': ['AdvancedMattesMutualInformation'],
    'Metric0Weight': ['0'],
    'Metric1Weight': ['1.000000'],
    'MovingImagePyramid': ["MovingSmoothingImagePyramid"],
    'NewSamplesEveryIteration': ['true'],
    'NumberOfHistogramBins': ['32.000000'],
    'NumberOfResolutions': ['4.000000'],
    'NumberOfSamplesForExactGradient': ['4096.000000'],
    'NumberOfSpatialSamples': ['2048.000000'],
    'Optimizer': ['AdaptiveStochasticGradientDescent'],
    'Registration': ['MultiMetricMultiResolutionRegistration'],
    'ResampleInterpolator': ['FinalBSplineInterpolator'],
    'Resampler': ['DefaultResampler'],
    'ResultImageFormat': ['nii'],
    "RequiredRatioOfValidSamples": ['0.05'],
    'Transform': ['BSplineTransform'],
    'WriteIterationInfo': ['false'],
    'WriteResultImage': ['true']
  }


DEFAULT_AFFINE_TRANSFORM = {
    'AutomaticScalesEstimation': ('True'),
    'CenterOfRotationPoint': ('0.0', '0.0', '0.0'), 
    'CompressResultImage': ('false',), 
    'DefaultPixelValue': ('0.000000',), 
    'FinalBSplineInterpolationOrder': ('3',),
    'FixedInternalImagePixelType': ('float',), 
    'Index': ('0', '0', '0'), 
    'NumberOfParameters': ('12',),  
    'ResampleInterpolator': ['FinalNearestNeighborInterpolator'], 
    'Resampler': ('DefaultResampler',), 
    'ResultImageFormat': ('nii',), 
    'ResultImagePixelType': ('float',), 
    'Transform': ('AffineTransform',),
    'UseDirectionCosines': ('true',)
    }





if __name__ == '__main__':
  start = time.time()
  run_amsaf()
  end = time.time()
  print("TIME: " + str(end-start))
