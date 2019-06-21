import SimpleITK as sitk

subject = "F"
trial = "20_90_p5"
trial_name = "trial" + trial + "_volume"

orig_image_path = "/media/jessica/Storage/Sub" + subject + "/US-mocap/" + trial_name + ".mha"
save_image_path = "/media/jessica/Storage/Sub" + subject + "/" + trial_name + ".nii"

img = sitk.ReadImage(orig_image_path)
sitk.WriteImage(img, save_image_path)