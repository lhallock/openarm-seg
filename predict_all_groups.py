import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append('src/')
import pipeline
import Unet
import logging
import time


# over_512_configs reference block: using this block will predict on all > 512 scans in the data set.

# over_512_configs = [("/media/jessica/Storage1/SubB/prediction_sources/over_512", "/media/jessica/Storage1/SubB/predictions/over_512", "/media/jessica/Storage1/SubB/all_nifti"),
# 					("/media/jessica/Storage1/SubC/prediction_sources/over_512", "/media/jessica/Storage1/SubC/predictions/over_512", "/media/jessica/Storage1/SubC/all_nifti"),
# 					("/media/jessica/Storage1/SubJ/prediction_sources/over_512", "/media/jessica/Storage1/SubJ/predictions/over_512", "/media/jessica/Storage1/SubJ/all_nifti"),
# 					("/media/jessica/Storage1/SubK/prediction_sources/over_512", "/media/jessica/Storage1/SubK/predictions/over_512", "/media/jessica/Storage1/SubK/all_nifti")]

# undeR_512_configs reference block: using this block will predict on all <= 512 scans in the data set.

# under_512_configs = [("/media/jessica/Storage1/SubA/prediction_sources/under_512", "/media/jessica/Storage1/SubA/predictions/under_512", "/media/jessica/Storage1/SubA/all_nifti"),
# 					 ("/media/jessica/Storage1/SubB/prediction_sources/under_512", "/media/jessica/Storage1/SubB/predictions/under_512", "/media/jessica/Storage1/SubB/all_nifti"),
# 					 ("/media/jessica/Storage1/SubC/prediction_sources/under_512", "/media/jessica/Storage1/SubC/predictions/under_512", "/media/jessica/Storage1/SubC/all_nifti"),
# 					 ("/media/jessica/Storage1/SubD/prediction_sources/under_512", "/media/jessica/Storage1/SubD/predictions/under_512", "/media/jessica/Storage1/SubD/all_nifti"),
# 					 ("/media/jessica/Storage1/SubE/prediction_sources/under_512", "/media/jessica/Storage1/SubE/predictions/under_512", "/media/jessica/Storage1/SubE/all_nifti"),
# 					 ("/media/jessica/Storage1/SubF/prediction_sources/under_512", "/media/jessica/Storage1/SubF/predictions/under_512", "/media/jessica/Storage1/SubF/all_nifti"),
# 					 ("/media/jessica/Storage1/SubG/prediction_sources/under_512", "/media/jessica/Storage1/SubG/predictions/under_512", "/media/jessica/Storage1/SubG/all_nifti"),
# 					 ("/media/jessica/Storage1/SubH/prediction_sources/under_512", "/media/jessica/Storage1/SubH/predictions/under_512", "/media/jessica/Storage1/SubH/all_nifti"),
# 					 ("/media/jessica/Storage1/SubI/prediction_sources/under_512", "/media/jessica/Storage1/SubI/predictions/under_512", "/media/jessica/Storage1/SubI/all_nifti"),
# 					 ("/media/jessica/Storage1/SubJ/prediction_sources/under_512", "/media/jessica/Storage1/SubJ/predictions/under_512", "/media/jessica/Storage1/SubJ/all_nifti"),
# 					 ("/media/jessica/Storage1/SubK/prediction_sources/under_512", "/media/jessica/Storage1/SubK/predictions/under_512", "/media/jessica/Storage1/SubK/all_nifti")]

over_512_configs = [("/media/jessica/Storage1/SubC/prediction_sources/over_512", "/media/jessica/Storage1/SubC/predictions/over_512", "/media/jessica/Storage1/SubC/all_nifti"),
					("/media/jessica/Storage1/SubK/prediction_sources/over_512", "/media/jessica/Storage1/SubK/predictions/over_512", "/media/jessica/Storage1/SubK/all_nifti")]

under_512_configs = [("/media/jessica/Storage1/SubC/prediction_sources/under_512", "/media/jessica/Storage1/SubC/predictions/under_512", "/media/jessica/Storage1/SubC/all_nifti"),
					 ("/media/jessica/Storage1/SubG/prediction_sources/under_512", "/media/jessica/Storage1/SubG/predictions/under_512", "/media/jessica/Storage1/SubG/all_nifti"),
					 ("/media/jessica/Storage1/SubH/prediction_sources/under_512", "/media/jessica/Storage1/SubH/predictions/under_512", "/media/jessica/Storage1/SubH/all_nifti"),
					 ("/media/jessica/Storage1/SubK/prediction_sources/under_512", "/media/jessica/Storage1/SubK/predictions/under_512", "/media/jessica/Storage1/SubK/all_nifti")]


# group_whitelist = ['group_1_1_final_sub', 'group_1_4_final_sub', 'group_1_5_final_sub',

# 		  		   'group_2_5_final_sub', 'group_2_6_final_sub',

# 		  		   'group_3_2_final_sub', 'group_3_3_final_sub', 'group_3_4_final_sub',

# 		  		   'group_4_4_final_sub', 'group_4_5_final_sub']

group_whitelist = ['G_augs_group', 'H_augs_group', 'C_augs_group', 'K_augs_group']


def main():
	args = sys.argv[1:]

	models_dir = args[0] if len(args) != 0 else "/media/jessica/Storage1/models/u-net_v1-0/"

	group_folders = []

	for model_folder in sorted(os.listdir(models_dir))[::-1]:
		print(model_folder)
		if os.path.isdir(os.path.join(models_dir, model_folder)) and 'group' in model_folder:
			group_folders.append(model_folder)

	print(group_folders)
	time.sleep(5)

	for group in group_folders:
		if group not in group_whitelist:
			print("skipped", group)
			continue

		print(group)
		for size in [512, 1024]:
			tf.reset_default_graph()
			sess = tf.Session()
			model = Unet.Unet(0, 0.5, 0.5, h = size, w = size) # Mostly arbitrary initialization with correct size
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()

			configs = under_512_configs if size == 512 else over_512_configs

			pipeline.load_model(models_dir, group, saver, sess)

			for config in configs:
				pipeline.predict_all_segs(config[0], config[1] + "/" + group, config[2], model, sess, reorient = True, predict_lower = False)




if __name__ == '__main__':
	logger = logging.getLogger('__name__')
	stream = logging.StreamHandler(stream=sys.stdout)
	stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
	logger.handlers = []
	logger.addHandler(stream)
	logger.setLevel(logging.DEBUG)
	main()