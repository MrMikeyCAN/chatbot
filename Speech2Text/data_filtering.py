import csv
import os
from sklearn.model_selection import train_test_split

data_list = []


def add_data(files, path, language):
    for file in files:
        data_list.append({
            'Path': os.path.join(path, file),
            'Language': language
        })


def process_folders(path):
    i = 0
    for root, dirs, files in os.walk(path):
        if 'clips' in dirs:
            clips_path = os.path.join(root, 'clips')
            clips_files = os.listdir(clips_path)
            add_data(clips_files, clips_path, i)
            i += 1


base_path = "Datasets"
process_folders(base_path)

train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

fields = ["Path", "Language"]
train_filename = "Datasets/train.csv"
test_filename = "Datasets/test.csv"

with open(train_filename, "w", encoding="utf8", newline='') as train_csv:
    writer = csv.DictWriter(train_csv, fieldnames=fields)
    writer.writeheader()
    writer.writerows(train_data)

with open(test_filename, "w", encoding="utf8", newline='') as test_csv:
    writer = csv.DictWriter(test_csv, fieldnames=fields)
    writer.writeheader()
    writer.writerows(test_data)
