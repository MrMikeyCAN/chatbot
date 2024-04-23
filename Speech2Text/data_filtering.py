## yorum satırları eklenecek (Tamamlandı)
import csv
import os
from sklearn.model_selection import train_test_split

data_list = []


def add_data(files, path, language):
    for file in files:
        data_list.append({
            'features': os.path.join(path, file),
            'labels': language
        })


def process_folders(path):
    i = 0
    for root, dirs, files in os.walk(path):
        if 'clips' in dirs:
            clips_path = os.path.join(root, 'clips')
            clips_files = os.listdir(clips_path)
            add_data(clips_files, clips_path, i)
            i += 1


def create_splits(data, test_size=0.2, val_size=0.1):

    train_data, temp_data = train_test_split(data, test_size=test_size + val_size, random_state=42)

    val_size_adjusted = val_size / (test_size + val_size)
    val_data, test_data = train_test_split(temp_data, test_size=val_size_adjusted, random_state=42)

    return train_data, val_data, test_data


base_path = "Datasets"
process_folders(base_path)

test_size = 0.2
val_size = 0.1

train_data, val_data, test_data = create_splits(data_list, test_size, val_size)

fields = ["features", "labels"]
train_filename = "Datasets/train.csv"
val_filename = "Datasets/val.csv"
test_filename = "Datasets/test.csv"

with open(train_filename, "w", encoding="utf8", newline='') as train_csv:
    writer = csv.DictWriter(train_csv, fieldnames=fields)
    writer.writeheader()
    writer.writerows(train_data)

with open(val_filename, "w", encoding="utf8", newline='') as val_csv:
    writer = csv.DictWriter(val_csv, fieldnames=fields)
    writer.writeheader()
    writer.writerows(val_data)

with open(test_filename, "w", encoding="utf8", newline='') as test_csv:
    writer = csv.DictWriter(test_csv, fieldnames=fields)
    writer.writeheader()
    writer.writerows(test_data)
