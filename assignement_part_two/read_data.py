import h5py
import os

def get_dataset_name (file_name_with_dir):
    filename_without_dir = file_name_with_dir.split ('/')[-1]
    temp = filename_without_dir. split ('_') [: -1]
    dataset_name ="_".join (temp)
    return dataset_name

def read_h5_file (filename_path):
    with h5py. File (filename_path, 'r') as f:
        dataset_name = get_dataset_name (filename_path)
        matrix = f.get (dataset_name) [()]
        return matrix
    
def load_data_from_h5_files(parent_directory="Final_project_data", subdirectory="Intra" , type_of_data = "Test"):
    directrory_path = os.path.join(parent_directory, type_of_data, subdirectory)
    data = []
    for filename in os.listdir (directrory_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(directrory_path, filename)
            file_path = file_path.replace ('\\', '/')
            matrix = read_h5_file (file_path)
            data.append(matrix)
    return data

if __name__ == "__main__":
    directory_path = "Final_project_data/Intra/train"
    data = load_data_from_h5_files (directory_path)
