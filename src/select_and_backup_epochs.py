import numpy as np
import random
import os
import sys
import itertools
from math import floor, ceil
import logging
import pickle
import time
import datetime
from shutil import copyfile

groups_dir = "/media/jessica/Storage1/models/u-net_v1-0/groups_final_sub"

group_whitelist = ['group_1_1_final_sub', 'group_1_4_final_sub', 'group_1_5_final_sub',

		  		   'group_2_5_final_sub', 'group_2_6_final_sub',

		  		   'group_3_2_final_sub', 'group_3_3_final_sub', 'group_3_4_final_sub',

		  		   'group_4_4_final_sub', 'group_4_5_final_sub']

desired_epochs = {
	"group_1_1_final_sub" : 17,
	"group_1_2" : 8,
	"group_1_3" : 10,
	"group_1_4_final_sub" : 38,
	"group_1_5_final_sub" : 28,
	 
	"group_2_1" : 7,
	"group_2_2" : 10,
	"group_2_3" : 8,
	"group_2_4" : 8,
	"group_2_5_final_sub" : 32,
	"group_2_6_final_sub" : 33,

	"group_3_1" : 6,
	"group_3_2_final_sub" : 39,
	"group_3_3_final_sub" : 30,
	"group_3_4_final_sub" : 30,

	"group_4_1" : 7,
	"group_4_2" : 10,
	"group_4_3" : 8,
	"group_4_4_final_sub" : 39,
	"group_4_5_final_sub" : 39,

	"group_5_1" : 10,
	"group_5_2" : 5,
	"group_5_3" : 5,
	"group_5_4" : 8,

	"group_6_1" : 8,
	"group_6_2" : 10,
	"group_6_3" : 12,
	"group_6_4" : 17,
	"group_6_5" : 12,

	"group_7_1" : 4,
	"group_7_2" : 9,
	"group_7_3" : 9,
	"group_7_4" : 11,

	"group_8_1" : 10,
	"group_8_2" : 11,
}


def main():
	for group_folder in sorted(os.listdir(groups_dir)):

		if group_folder not in group_whitelist:
			print("skipped", group_folder)
			continue

		print("current:", group_folder)

		group_model_dir = os.path.join(groups_dir, group_folder)

		for item in os.listdir(group_model_dir):
			if not os.path.isdir(item) and item.startswith(group_folder + "data"):
				os.remove(os.path.join(group_model_dir, item))
			if not os.path.isdir(item) and item == (group_folder + ".index"):
				os.remove(os.path.join(group_model_dir, item))
			if not os.path.isdir(item) and item == (group_folder + ".meta"):
				os.remove(os.path.join(group_model_dir, item))

		data_target = group_folder + ".data-00000-of-00001"
		index_target = group_folder + ".index"
		meta_target = group_folder + ".meta"

		desired_epoch_folder = group_folder + "_epoch_" + str(desired_epochs[group_folder])

		print("\tSelecting epoch", desired_epochs[group_folder], "looking in:", desired_epoch_folder)

		data_target_path = os.path.join(os.path.join(group_model_dir, desired_epoch_folder), data_target)
		index_target_path = os.path.join(os.path.join(group_model_dir, desired_epoch_folder), index_target)
		meta_target_path = os.path.join(os.path.join(group_model_dir, desired_epoch_folder), meta_target)

		new_data_path = os.path.join(group_model_dir, data_target)
		new_index_path = os.path.join(group_model_dir, index_target)
		new_meta_path = os.path.join(group_model_dir, meta_target)

		copyfile(data_target_path, new_data_path)
		copyfile(index_target_path, new_index_path)
		copyfile(meta_target_path, new_meta_path)


if __name__ == '__main__':
	logger = logging.getLogger('__name__')
	stream = logging.StreamHandler(stream=sys.stdout)
	stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
	logger.handlers = []
	logger.addHandler(stream)
	logger.setLevel(logging.DEBUG)
	main()