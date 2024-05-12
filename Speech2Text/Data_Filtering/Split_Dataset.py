import csv
import os
from sklearn.model_selection import train_test_split
import json

filetype = ".csv"
datatypes = [".wav", ".mp3", ".flac"]

# Reading json file
try:
    with open('Parameters/split.json', 'r', encoding='utf-8') as f:
        json_file = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Json File Not Found.")

# Play all
for model in json_file:
    data_list = list()
    dataset = list()

    try:
        # Parameters
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
    filenames = [train_filename, test_filename]

    # Process
    def process_folders(path, label):
        files = os.listdir(path)
        for file in files:
            if file.endswith(datatypes[datatype_index]):
                data_list.append({
                    fields[0]: os.path.join(path, file),
                    fields[1]: label
                })

    for paths, labels in labels.items():
        try:
            file_path = os.path.join(root_dir, paths)
            process_folders(file_path, labels)
        except AssertionError:
            raise AssertionError("Path Not Found")

    if not train_only:
        # Split
        if len(data_list) is not 0:
            train_data, test_data = train_test_split(data_list, test_size=split_value_1, random_state=random_seed)
            if has_val:
                test_data, val_data = train_test_split(test_data, test_size=split_value_2, random_state=random_seed)
                filenames.append(val_filename)
                dataset.append(val_data)

            dataset.append(train_data)
            dataset.append(test_data)

            # Write
            for filename, data in zip(filenames, dataset):
                file_path = os.path.join(root_dir, filename)
                if not write_on_it:
                    count = 0
                    start = file_path
                    while os.path.exists(start):
                        start = file_path[: -len(datatypes[datatype_index])] + str(count) + filetype
                        count += 1
                    file_path = start

                with open(file_path, "w", encoding="utf8", newline='') as _file:
                    writer = csv.DictWriter(f=_file, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(data)
        else:
            raise ValueError("No Data.")
    else:
        # Just write
        with open(os.path.join(root_dir, train_filename), "w", encoding="utf8", newline='') as _file:
            writer = csv.DictWriter(f=_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data_list)
