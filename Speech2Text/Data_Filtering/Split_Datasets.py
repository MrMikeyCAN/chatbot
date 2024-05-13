import time
import warnings
import datasets
from sklearn.model_selection import train_test_split
import json
import csv
import os

warnings.warn("Speech to Text name must be stt")
warnings.warn("VAD name must be vad")

time.sleep(0.1)

# Set types
dataset = list()
filetype = ".csv"
datatypes = [".wav", ".mp3", ".flac"]

# Reading json file
try:
    with open('data.json', 'r', encoding='utf-8') as f:
        json_file = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Json File Not Found.")


# Write function
def write(dir_path: str, filename: str, w_data: list, count: int = 0):
    path = os.path.join(dir_path, filename)
    if not write_on_it:
        start = path
        while os.path.exists(start):
            start = path[: -len(datatypes[datatype_index])] + str(count) + filetype
            count += 1
        path = start

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

    model_name = json_file[model]["name"].lower()

    if model_name.lower() == 'stt':
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
            train_filename = json_file[model]["train_filename"]
            val_filename = json_file[model]["val_filename"]
            test_filename = json_file[model]["test_filename"]
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
        assert isinstance(train_filename, str), "train_filename must be str"
        assert isinstance(val_filename, str), "val_filename must be str"
        assert isinstance(test_filename, str), "test_filename must be str"

        assert os.path.exists(root_dir), "root_dir must be an existing directory"

        for key in labels.keys():
            assert os.path.exists(os.path.join(root_dir, key)), f"{key} already not exists"

        train_filename += filetype
        val_filename += filetype
        test_filename += filetype

        filenames = [train_filename, val_filename, test_filename]

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
                write(os.path.join(root_dir, label_data), name, data)
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
            train_filename = json_file[model]['train_filename']
            val_filename = json_file[model]['val_filename']
            test_filename = json_file[model]['test_filename']
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
        assert isinstance(train_filename, str), "train_filename must be str"
        assert isinstance(val_filename, str), "val_filename must be str"
        assert isinstance(test_filename, str), "test_filename must be str"

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
        train_filename += filetype
        val_filename += filetype
        test_filename += filetype

        dataset = process_folders(root_dir, labels, datatype_index)

        if not train_only:
            # Split
            if len(dataset) is not 0:
                train_data, test_data = train_test_split(dataset, test_size=split_value_1, random_state=random_seed)
                if has_val:
                    test_data, val_data = train_test_split(test_data, test_size=split_value_2, random_state=random_seed)
                    write(root_dir, val_filename, val_data)

                write(root_dir, train_filename, train_data)
                write(root_dir, test_filename, test_data)
            else:
                raise ValueError("No Data.")
        else:
            write(root_dir, train_filename, dataset)
