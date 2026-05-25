import h5py
import os

labels = {
    "rest": 0,
    "motor": 1,
    "math": 2,
    "memory": 3
}

class DataSet:
    def __init__(self, persons, matrixes_data, matrixes_label):
        # ignore for now, but we can use it later to do person-specific statistical analysis (maybe out of scope)
        self.persons = persons

        self.x_data = matrixes_data
        self.y_data = matrixes_label
        self.labels = labels

class PersonData:
    def __init__(self, id):
        self.id = id
        self.tasks = []

    def add_task(self, task_data):
        self.tasks.append(task_data)

    def get_tasks(self):
        return self.tasks
    
    def get_tasks_by_name(self, task_name):
        return [task for task in self.tasks if task.name == task_name]
    
    def get_task_by_id(self, task_id):
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

class TaskData:
    def __init__(self, name, id, matrix):
        self.name = name
        self.id = id
        self.matrix = matrix

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
    directrory_path = os.path.join(parent_directory, subdirectory, type_of_data)
    directrory_path = directrory_path.replace ('\\', '/')

    persons = []
    x_data = []
    y_data = []

    for filename in os.listdir(directrory_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(directrory_path, filename)
            file_path = file_path.replace ('\\', '/')
            task_name, person_id, task_id = retrieve_file_name_info(filename)
            matrix = read_h5_file (file_path)

            x_data.append(matrix)
            y_data.append(task_name)

            task_data = TaskData(name=task_name, id=task_id, matrix=matrix)

            if not any(person.id == person_id for person in persons):
                new_person = PersonData(id=person_id)
                persons.append(new_person)

            person = next(person for person in persons if person.id == person_id)
            person.add_task(task_data)
    return persons, x_data, y_data

if __name__ == "__main__":
    subdirectory = "Cross"
    type_of_data = "test3"
    persons, x_data, y_data = load_data_from_h5_files(subdirectory=subdirectory, type_of_data=type_of_data)
    print (f"Number of persons loaded: {len(persons)}")
    print (f"Number of tasks for first person: {len(persons[0].get_tasks())}")
    print (f"Data shape: {x_data[0].shape}")