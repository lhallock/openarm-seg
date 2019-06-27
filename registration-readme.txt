HART Lab - Registration code

Description:
	Code to generate the transformation between two images and use it to map the segmentation of one image onto the other

Dependencies:
	SimpleElastix
	Numpy

How to run:
	Simply call `python registration.py`
	Edit the "run()" method in the file to change the parameters.

Parameters:
	verbose: Boolean determining verbosity of output
    
    Input images: For all of these, make sure the variables are set to `read_image("filename_as_string")`. This code has been written under the assumption that the file is a NifTI file (.nii)
	    unsegmented_image
	    segmented_image
	    segmentation
	 
	new_segmentation: String representing location to save the output segmentation

	Affine pre-registration parameters: There can be issues registering images that are too "far apart." For this case, we perform an affine transformation on the segmentation and segmented image to get the images "closer" to each other. This transform works such that x' = Ax + t. Let A be a 3x3 numpy array and t be a 1x3 numpy array

		Example affine pre-registration parameters:
		    A = np.array([[1.0, 0, 0],
		                    [0, 1.0, 0], 
		                    [0, 0.0, 1.0]])
		    t = np.array([[0, 0, 0]])