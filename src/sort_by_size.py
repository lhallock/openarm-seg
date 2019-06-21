import nibabel as nib
import os

target_dir = "/media/jessica/Storage/SubK"
over_dir = os.path.join(target_dir, "over_512")
under_dir = os.path.join(target_dir, "under_512")

for file in os.listdir(target_dir):
	if 'trial' in file and 'volume' in file and file.endswith('.nii'):
		nifti = nib.load(os.path.join(target_dir, file))
		nifti_shape = nifti.get_fdata().shape
		sorted_shape = sorted(nifti_shape, reverse=True)[1:]
		if sorted_shape[0] <= 512 and sorted_shape[1] <= 512:
			os.rename(os.path.join(target_dir, file), os.path.join(under_dir, file))
		else:
			os.rename(os.path.join(target_dir, file), os.path.join(over_dir, file))

