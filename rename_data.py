"""
Prepend zeros to the number in filenames of processed scan data.

Usage: python rename_data.py [data_directory]

.
├── data_directory
│   ├── 0_label.png
│   └── 0_raw.npz
│   └── ...

"""


import os
from sys import argv

if __name__ == "__main__":
	source_dir = argv[1]
	for file_name in os.listdir(source_dir):
		if not file_name.startswith('.'):
			file_parts = file_name.split("_")
			orig_path = os.path.join(source_dir, file_name)
			new_name = file_parts[0].zfill(5) + "_" + file_parts[1]
			new_path = os.path.join(source_dir, new_name)
			os.rename(orig_path, new_path)

