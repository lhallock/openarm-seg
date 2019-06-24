import SimpleITK as sitk
import os


target_directory = "/media/jessica/Storage/SubF";

for file in os.listdir(os.path.join(target_directory, 'US-mocap')):
	if 'volume' in file and file.endswith('.mha'):
		orig_image_path = os.path.join(os.path.join(target_directory, 'US-mocap'), file)

		new_file_name = file.split('.mha')[0] + '.nii'
		save_image_path = os.path.join(target_directory, new_file_name)

		print("Attempting to convert", new_file_name)

		try:
			img = sitk.ReadImage(orig_image_path)
			sitk.WriteImage(img, save_image_path)
		except Exception as e:
			print(e)
			print("Failed to convert", new_file_name)
			continue