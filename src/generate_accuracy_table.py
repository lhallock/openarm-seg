import numpy as np
import random
import os
import sys
import itertools
sys.path.append('src/')
import nibabel as nib
from math import floor, ceil
import pipeline
import Unet
import logging
import pickle
import time
import datetime
import nibabel as nib
from prettytable import PrettyTable
import sys


base_path = "/media/jessica/Storage1/"

subjects = ["B", "F", "G", "H"]

size_dirs = ["under_512", "over_512"]

cols = ['Group', 'trial1B', 'trial2B', 'trial3B', 'trial4B', 'trial5B', 'trial6B', 'trial6F', 'trial6G', 'trial6H',
		'trial7B', 'trial7H', 'trial8B', 'trial8H', 'trial9B', 'trial9H', 'trial10B', 'trial10H', 'trial11B',
		'trial12B', 'trial13B', 'trial14B', 'trial15B', 'trial16B', 'trial17B', 'trial18B', 'trial19B', 'trial20B']


trial_mapping =	{'trial1B' : 1,
				 'trial2B' : 2,
				 'trial3B' : 3,
				 'trial4B' : 4,
				 'trial5B' : 5,
				 'trial6B' : 6,
				 'trial6F' : 7,
				 'trial6G' : 8,
				 'trial6H' : 9,
				 'trial7B' : 10,
				 'trial7H' : 11,
				 'trial8B' : 12,
				 'trial8H' : 13,
				 'trial9B' : 14,
				 'trial9H' : 15,
				 'trial10B' : 16,
				 'trial10H' : 17,
				 'trial11B' : 18,
				 'trial12B' : 19,
				 'trial13B' : 20,
				 'trial14B' : 21,
				 'trial15B' : 22,
				 'trial16B' : 23,
				 'trial17B' : 24,
				 'trial18B' : 25,
				 'trial19B' : 26,
				 'trial20B' : 27}

groups = 		  ['group_1_1_final_sub', 'group_1_4_final_sub', 'group_1_5_final_sub',

		  		   'group_2_5_final_sub', 'group_2_6_final_sub',

		  		   'group_3_2_final_sub', 'group_3_3_final_sub', 'group_3_4_final_sub',

		  		   'group_4_4_final_sub', 'group_4_5_final_sub']


def main():
	table = PrettyTable()

	table.field_names = cols

	table_data = []

	for group in groups:
		table_data.append([group] + ([0] * 27))

	for sub in subjects:
		for size_dir in size_dirs:
			groups_path = base_path + "Sub" + sub + "/predictions/" + size_dir
			groups_with_preds = sorted(os.listdir(groups_path))[::-1]

			ground_truth_path = base_path + "Sub" + sub + "/prediction_sources/" + size_dir

			print(groups_path)
			print(ground_truth_path)
			for curr_group in groups_with_preds:
				if not curr_group in groups:
					print("\tSkipped", curr_group)
					continue

				print("\t", curr_group)
				predictions_path = os.path.join(groups_path, curr_group)
				for pred_seg_name in os.listdir(predictions_path):
					print("\t\t", pred_seg_name, end=' ')

					prediction_data = nib.load(os.path.join(predictions_path, pred_seg_name)).get_fdata()
					prediction_data = np.swapaxes(prediction_data, 0, 2)
					prediction_data = prediction_data[prediction_data.shape[0]-650:]
					prediction_data = np.rint(prediction_data)

					ground_truth_data = None

					trial_name = pred_seg_name.split("_")[0]
					trial_num = trial_name[5:]

					# Find matching ground truth
					for trial_dir in os.listdir(ground_truth_path):
						if trial_name in trial_dir:
							trial_path = os.path.join(ground_truth_path, trial_dir)
							for nifti_name in os.listdir(trial_path):
								if 'seg' in nifti_name:
									target_ground_truth = nifti_name

									ground_truth_data = nib.load(os.path.join(trial_path, target_ground_truth)).get_fdata()
									ground_truth_data = np.swapaxes(ground_truth_data, 0, 2)
									ground_truth_data = ground_truth_data[ground_truth_data.shape[0]-650:]
									ground_truth_data = np.rint(ground_truth_data)


					acc_val = 0
					acc_sel = sys.argv[1]
					if acc_sel == 'mean_iou':
						acc_val = average_iou_accuracy(prediction_data, ground_truth_data)
					elif acc_sel == 'total_percent':
						acc_val = percent_correct(prediction_data, ground_truth_data)
					elif acc_sel == 'bicep_iou':
						acc_val = bicep_iou(prediction_data, ground_truth_data)
					elif acc_sel == 'bicep_percent':
						acc_val = bicep_percent(prediction_data, ground_truth_data)
					elif acc_sel == 'humerus_iou':
						acc_val = humerus_iou(prediction_data, ground_truth_data)
					elif acc_sel == 'humerus_percent':
						acc_val = humerus_percent(prediction_data, ground_truth_data)
					else:
						raise ValueError('Invalid accuracy metric selection.')

					print(acc_val)

					# Find place to put accuracy value in table
					target_row = 0
					target_col = trial_mapping[trial_name + sub]
					for row in range(len(table_data)):
						if table_data[row][0] == curr_group:
							target_row = row
							break

					table_data[target_row][target_col] = acc_val



	for row in table_data:
		table.add_row(row)

	save_table(table, table_data, acc_sel)


def save_table(table, table_data, metric):
	table_str = table.get_string()
	save_name = "accuracy_table_" + metric + datetime.datetime.now().isoformat()

	with open(save_name,'w') as file:
		file.write(table_str)

	with open(save_name + "_data.pickle", 'wb') as handle:
		pickle.dump(table_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def bicep_percent(prediction, reference):
	total_bicep = np.sum(reference == 52)
	prediction[prediction != 52] = 0
	reference[reference != 52] = 0

	return np.sum(np.logical_and(prediction, reference)) / total_bicep


def bicep_iou(prediction, reference):
    prediction_52 = (prediction == 52) * 52
    reference_52 = (reference == 52) * 52

    iou_52 = iou_accuracy(prediction_52, reference_52)

    return iou_52


def humerus_percent(prediction, reference):
	total_humerus = np.sum(reference == 7)
	prediction[prediction != 7] = 0
	reference[reference != 7] = 0

	return np.sum(np.logical_and(prediction, reference)) / total_humerus

def humerus_iou(prediction, reference):
    prediction_7 = (prediction == 7) * 7
    reference_7 = (reference == 7) * 7

    iou_7 = iou_accuracy(prediction_7, reference_7)

    return iou_7

def percent_correct(prediction, reference):
    return np.sum(prediction == reference) / reference.size

def iou_accuracy(prediction, reference):
    intersection = np.logical_and(prediction, reference)
    union = np.logical_or(prediction, reference)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def average_iou_accuracy(prediction, reference):
    prediction_7 = (prediction == 7) * 7
    reference_7 = (reference == 7) * 7
    
    prediction_52 = (prediction == 52) * 52
    reference_52 = (reference == 52) * 52
    
    iou_7 = iou_accuracy(prediction_7, reference_7)
    iou_52 = iou_accuracy(prediction_52, reference_52)
    average_iou = (iou_7 + iou_52) / 2
    
    return average_iou

if __name__ == '__main__':
	logger = logging.getLogger('__name__')
	stream = logging.StreamHandler(stream=sys.stdout)
	stream.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
	logger.handlers = []
	logger.addHandler(stream)
	logger.setLevel(logging.DEBUG)
	main()