import h5py
import os
import numpy as np

labels = {
    "rest": 0,
    "motor": 1,
    "math": 2,
    "memory": 3
}

class DataSet:
    def __init__(self, persons, matrixes_data, matrixes_label):
        # ignore for now, but we can use it later to do person-specific statistical analysis (maybe out of scope)
        self.persons: list[PersonData] = persons

        self.x_data: list[np.ndarray] = matrixes_data
        self.y_data: list[str] = matrixes_label
        self.labels : dict[str, int] = labels

class ScanData:
    def __init__(self, id: str, task: str, matrix: np.ndarray):
        self.id: str = id
        self.task: str = task
        self.matrix: np.ndarray = matrix

class PersonData:
    def __init__(self, id: str):
        self.id: str = id
        self.scans: list[ScanData] = []

    def add_scan(self, scan_data: ScanData):
        self.scans.append(scan_data)

    def get_scans(self):
        return self.scans
    
    def get_scans_by_task_name(self, task_name: str):
        return [scan for scan in self.scans if scan.task == task_name]
    
    def get_scan_by_id(self, scan_id: str):
        for scan in self.scans:
            if scan.id == scan_id:
                return scan
        return None
    
    def get_tasks_names(self):
        scans = self.get_scans()
        tasks_names = []
        for scan in scans: 
            tasks_names.append(scan.get_task_name())
        return tasks_names
    
    def get_task_name(self): #returns alphabetic name of task
        return next(key for key, value in labels.items() if value == self.task)

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_') [: -1]
    dataset_name ="_".join(temp)
    return dataset_name

def read_h5_file (filename_path):
    with h5py. File (filename_path, 'r') as f:
        dataset_name = get_dataset_name(filename_path)
        matrix = f.get (dataset_name)[()]
        return matrix
    
def retrieve_file_name_info(file_path: str):
    task_name = None
    person_id = None
    task_id = None
    if file_path.find("rest") != -1:
        task_name = labels["rest"]
        person_id = file_path.split('_')[1]
        task_id = file_path.split('_')[2].split('.')[0]
    elif file_path.find("motor") != -1:
        task_name = labels["motor"]
        person_id = file_path.split('_')[2]
        task_id = file_path.split('_')[3].split('.')[0]
    elif file_path.find("math") != -1:
        task_name = labels["math"]
        person_id = file_path.split('_')[3]
        task_id = file_path.split('_')[4].split('.')[0]
    elif file_path.find("memory") != -1:
        task_name = labels["memory"]
        person_id = file_path.split('_')[3]
        task_id = file_path.split('_')[4].split('.')[0]


    return task_name, person_id, task_id

def load_data_from_h5_files(parent_directory="Final_project_data", subdirectory="Intra" , type_of_data = "test"):
    # Returns:
    # - persons : an array containing person ids
    directrory_path = os.path.join(parent_directory, subdirectory, type_of_data)
    directrory_path = directrory_path.replace ('\\', '/')

    persons = []

    for filename in os.listdir(directrory_path):
        if filename.endswith('.h5'):
            person_id, scan_data = load_scan(directrory_path, filename)

            if not any(person.id == person_id for person in persons):
                new_person = PersonData(id=person_id)
                persons.append(new_person)

            person = next(person for person in persons if person.id == person_id)
            person.add_scan(scan_data)
    return  persons

def load_data_from_h5_file(parent_directory="Final_project_data", subdirectory="Intra" , type_of_data = "test", filename = "rest_1_1.h5"):
    directrory_path = os.path.join(parent_directory, subdirectory, type_of_data)
    directrory_path = directrory_path.replace ('\\', '/')
    file_path = os.path.join(directrory_path, filename)
    file_path = file_path.replace ('\\', '/')
    task_name, person_id, task_id = retrieve_file_name_info(filename)
    matrix = read_h5_file (file_path)

    scan_data = ScanData(id=task_id, task=task_name , matrix=matrix)

    person = PersonData(id=person_id)
    person.add_scan(scan_data)

    return person

def load_scan(directrory_path, filename):
    file_path = os.path.join(directrory_path, filename)
    file_path = file_path.replace ('\\', '/')
    task_name, person_id, task_id = retrieve_file_name_info(filename)
    matrix = read_h5_file (file_path)

    scan_data = ScanData(id=task_id, task=task_name , matrix=matrix)
    return person_id,scan_data

if __name__ == "__main__":
    subdirectory = "Intra"
    type_of_data = "train"
    persons, x_data, y_data = load_data_from_h5_files(subdirectory=subdirectory, type_of_data=type_of_data)
    print (f"Number of persons loaded: {len(persons)}")
    print (f"Number of scans for first person: {len(persons[0].get_scans())}")
    print(f"Name of tasks for first person:{persons[0].get_tasks_names()}")
    print (f"Data shape: {x_data[0].shape}")
    print(y_data)