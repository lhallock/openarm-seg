import numpy as np
import random
import os
import sys
import itertools
sys.path.append('src/')
from math import floor, ceil
import logging
import pickle
import time
import datetime
import ast


groups_dir = "/media/jessica/Storage1/models/u-net_v1-0/groups_17extra"


group_whitelist = ['group_1_3',

		  		   'group_2_2', 'group_2_3', 'group_2_4',

		  		   'group_3_2', 'group_3_3', 'group_3_4',

		  		   'group_4_2', 'group_4_3']


def main():

	output = open("all_model_info_" + datetime.datetime.now().isoformat(), 'w')

	for group_folder in sorted(os.listdir(groups_dir)):
			if group_folder not in group_whitelist:
				print("skipped", group_folder)
				continue

			output.write("########\n")
			output.write(group_folder + "\n")
			output.write("########\n")
			output.write('\n')

			print(group_folder)

			group_path = os.path.join(groups_dir, group_folder)
			losses_path = os.path.join(group_path, group_folder + "_losses")
			test_acc_path = os.path.join(group_path, group_folder + "_test_acc")
			val_accs_path = os.path.join(group_path, group_folder + "_val_accs")

			losses_file = open(losses_path, 'r')
			test_acc_file = open(test_acc_path, 'r')
			val_accs_file = open(val_accs_path, 'r')

			losses = losses_file.read().splitlines()
			test_acc = test_acc_file.read().splitlines()
			val_accs = val_accs_file.read().splitlines()

			losses_file.close()
			test_acc_file.close()
			val_accs_file.close()

			output.write(group_folder.upper() + " LOSSES (by epoch)\n")
			for i in range(len(losses)):
				losses[i] = ast.literal_eval(losses[i])
				output.write(str(i) + '\t' + str(losses[i]) + "\n")

			output.write('\n')

			output.write(group_folder.upper() + " TEST ACCS (final epoch)\n")
			for i in range(len(test_acc)):
				test_acc[i] = ast.literal_eval(test_acc[i])
				output.write(str(test_acc[i]) + "\n")

			output.write('\n')

			for i in range(len(val_accs)):
				val_accs[i] = ast.literal_eval(val_accs[i])

			output.write(group_folder.upper() + " VAL ACCS 1 (by epoch)\n")
			for i in range(len(val_accs)):
				output.write(str(i) + '\t' + str(val_accs[i][0]) + "\n")

			output.write('\n')

			output.write(group_folder.upper() + " VAL ACCS 2 (by epoch)\n")
			for i in range(len(val_accs)):
				output.write(str(i) + '\t' + str(val_accs[i][5]) + "\n")

			output.write('\n')




if __name__ == '__main__':
	logger = logging.getLogger('__name__')
	stream = logging.StreamHandler(stream=sys.stdout)
	stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
	logger.handlers = []
	logger.addHandler(stream)
	logger.setLevel(logging.DEBUG)
	main()