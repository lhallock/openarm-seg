source_dir = "/Users/nozik/Documents/HARTresearch/test_rename"
import os 

if __name__ == "__main__":
	for file_name in os.listdir(source_dir):
		if not file_name.startswith('.'):
			file_parts = file_name.split("_")
			orig_path = os.path.join(source_dir, file_name)
			new_name = file_parts[0].zfill(3) + "_" + file_parts[1]
			new_path = os.path.join(source_dir, new_name)
			os.rename(orig_path, new_path)

