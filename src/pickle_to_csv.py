import csv
from six.moves import cPickle as pickle
import numpy as np
import sys


def main(path_pickle,dest):
	print(path_pickle)
	print(dest)
	conv = dest + ".csv"
	print(conv)
	x = []
	header = ['Group','trial1B', 'trial2B', 'trial3B', 'trial4','trial5B','trial6B','trial6F','trial6G','trial6H','trial7B','trial7H','trial8B','trial8H','trial9B','trial9H','trial10B','trial10H','trial11B','trial12B','trial13B','trial14B','trial15B','trial16B','trial17B','trial18B','trial19B','trial20B']
	with open(path_pickle,'rb') as f:
		x = pickle.load(f)

	with open(conv,"a+") as f:
		writer = csv.writer(f,delimiter=',')
		writer.writerow(header)
		for line in x: writer.writerow(line)


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])
