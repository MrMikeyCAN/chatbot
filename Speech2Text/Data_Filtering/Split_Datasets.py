import warnings
import datasets
from sklearn.model_selection import train_test_split
import json
import csv
import os

# Close Warnings
warnings.filterwarnings("ignore")

# Set types
dataset = list()
datatypes = [".wav", ".mp3", ".flac"]

# Reading json file
try:
    with open('package.json', 'r', encoding='utf-8') as f:
        json_path = json.load(f)["data_filtering_json_path"]

    with open(json_path, 'r', encoding='utf-8') as f:
        json_file = json.load(f)

except FileNotFoundError:
    raise FileNotFoundError("Json File Not Found.")


# Write function
def write(dir_path: str, filename: str, w_data: list, name_index: int, count: int):
    global json_file, json_path

    path = os.path.join(dir_path, filename)
    if not write_on_it:
        while os.path.exists(path):
            filename = filename[:filename.rfind('.')]+str(count)+filename[filename.rfind('.'):]
            path = os.path.join(dir_path, filename)
            count += 1

        with open(json_path, 'w', encoding='utf-8') as file:
            json_file["filenames"][name_index] = filename
            json.dump(json_file, file, indent=4)

    with open(path, "w", encoding="utf8", newline='') as _file:
        writer = csv.DictWriter(f=_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(w_data)


# Prepare Dataset
def pre_dataset(features: list, data_labels: list) -> list:
    datalist = list()
    for data_feature, data_label in zip(features, data_labels):
        datalist.append({
            fields[0]: data_feature,
            fields[1]: data_label
        })
    return datalist


# Process Folders
def process_folders(root: str, data_labels: dict, index: int) -> list:
    datalist = list()
    for path, data_label in data_labels.items():
        try:
            file_path = os.path.join(root, path)
            files = os.listdir(file_path)
            for file in files:
                if file.endswith(datatypes[index]):
                    datalist.append({
                        fields[0]: os.path.join(file_path, file),
                        fields[1]: data_label
                    })
        except AssertionError:
            raise AssertionError("Path Not Found")
    return datalist


# Play all
for model in json_file:
    dataset.clear()

    if model == 'STT':
        try:
            # Get Parameters
            train_only = json_file[model]["train_only"]
            has_val = json_file[model]["has_val"]
            write_on_it = json_file[model]["write_on_it"]
            root_dir = json_file[model]["root_dir"]
            dataset_name = json_file[model]["dataset"]
            labels = json_file[model]["labels"]
            features_name = json_file[model]["features_name"]
            labels_name = json_file[model]["labels_name"]
            filenames = json_file[model]["filenames"]
            start_count = json_file[model]["start_count"]
        except KeyError:
            raise KeyError("Please Control Json File.")

        # Controls
        assert isinstance(train_only, bool), "train_only must be bool"
        assert isinstance(has_val, bool), "has_val must be bool"
        assert isinstance(write_on_it, bool), "write_on_it must be bool"
        assert isinstance(root_dir, str), "root_dir must be str"
        assert isinstance(dataset_name, str), "dataset_name must be str"
        assert isinstance(labels, dict), "labels must be dict"
        assert isinstance(features_name, str), "features_name must be str"
        assert isinstance(labels_name, str), "labels_name must be str"
        assert isinstance(filenames, list), "filenames must be list"
        assert isinstance(filenames[0], str), "filename_train must be str"
        assert isinstance(filenames[1], str), "filename_val must be str"
        assert isinstance(filenames[2], str), "filename_test must be str"
        assert isinstance(start_count, int), "start_count must be int"

        assert os.path.exists(root_dir), "root_dir must be an existing directory"

        for key in labels.keys():
            assert os.path.exists(os.path.join(root_dir, key)), f"{key} already not exists"

        start_count = max(start_count, 0)

        # Create Data
        for label_data in labels:
            dataset.clear()
            position = labels[label_data][0]
            mission = labels[label_data][1]
            data = datasets.load_dataset(dataset_name, position, data_dir=str(os.path.join(root_dir, label_data)))

            train_features = data["train"]["file"]
            train_labels = data["train"][mission]

            dataset.append(pre_dataset(train_features, train_labels))

            if not train_only:
                test_features = data["test"]["file"]
                test_labels = data["test"][mission]
                dataset.append(pre_dataset(test_features, test_labels))
                if has_val:
                    val_features = data["test"]["file"]
                    val_labels = data["test"][mission]
                    dataset.append(pre_dataset(val_features, val_labels))

            for data, name in zip(dataset, filenames):
                write(os.path.join(root_dir, label_data), name, data, name_index=filenames.index(name), count=start_count)
    else:
        try:
            # Get Parameters
            train_only = json_file[model]['train_only']
            has_val = json_file[model]['has_val']
            write_on_it = json_file[model]['write_on_it']
            split_value_1 = json_file[model]['split_value_1']
            split_value_2 = json_file[model]['split_value_2']
            random_seed = json_file[model]['random_seed']
            features_name = json_file[model]['features_name']
            labels_name = json_file[model]['labels_name']
            labels = json_file[model]['labels']
            type_index = json_file[model]['type_index']
            root_dir = json_file[model]['root_dir']
            filenames = json_file[model]["filenames"]
            start_count = json_file[model]["start_count"]
        except KeyError:
            raise KeyError("Please Control Json File.")

        # Controls
        assert isinstance(train_only, bool), "train_only must be bool"
        assert isinstance(has_val, bool), "has_val must be bool"
        assert isinstance(write_on_it, bool), "write_on_it must be bool"
        assert isinstance(split_value_1, float), "split_value_1 must be float"
        assert isinstance(split_value_2, float), "split_value_2 must be float"
        assert isinstance(random_seed, int), "random_seed must be int"
        assert isinstance(features_name, str), "features_name must be str"
        assert isinstance(labels_name, str), "labels_name must be str"
        assert isinstance(labels, dict), "labels must be dict"
        assert isinstance(type_index, int), "type_index must be int"
        assert isinstance(root_dir, str), "root_dir must be str"
        assert isinstance(filenames, list), "filenames must be list"
        assert isinstance(filenames[0], str), "filename_train must be str"
        assert isinstance(filenames[1], str), "filename_val must be str"
        assert isinstance(filenames[2], str), "filename_test must be str"
        assert isinstance(start_count, int), "start_count must be int"

        assert 0 <= split_value_1 <= 1, "split_value_1 should be between 0 and 1"
        assert 0 <= split_value_2 <= 1, "split_value_2 should be between 0 and 1"
        assert 0 <= random_seed <= 2**32-1, "Seed must be between 0 and 2**32 - 1"

        assert os.path.exists(root_dir), "root_dir must be an existing directory"

        for key in labels.keys():
            assert os.path.exists(os.path.join(root_dir, key)), f"{key} already not exists"

        if type_index >= len(datatypes):
            datatype_index = 0
        elif type_index <= 0:
            datatype_index = 0
        else:
            datatype_index = type_index

        fields = [features_name, labels_name]
        dataset = process_folders(root_dir, labels, datatype_index)
        start_count = max(start_count, 0)

        if not train_only:
            # Split
            if len(dataset) is not 0:
                train_data, test_data = train_test_split(dataset, test_size=split_value_1, random_state=random_seed)
                if has_val:
                    test_data, val_data = train_test_split(test_data, test_size=split_value_2, random_state=random_seed)
                    write(root_dir, filenames[1], val_data, name_index=1, count=start_count)

                write(root_dir, filenames[0], train_data, name_index=0, count=start_count)
                write(root_dir, filenames[2], test_data, name_index=2, count=start_count)
            else:
                raise ValueError("No Data.")
        else:
            write(root_dir, filenames[0], dataset, name_index=0, count=start_count)
