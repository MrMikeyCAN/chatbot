# yorum satırları eklenecek (Tamamlandı)

import random
from pydub import AudioSegment
import csv
import os

datasets_path = "../Datasets/Filter_Datasets/"


def add_data(feature_path, label_path, export_path):
    clean = os.listdir(feature_path)
    noise = os.listdir(label_path)
    datasets = []
    for feature in clean:
        if feature.endswith(".wav"):
            audio_clean = AudioSegment.from_file(os.path.join(feature_path, feature))
            audio_noise = AudioSegment.from_file(os.path.join(label_path, random.choice(noise)))

            combined = audio_clean.overlay(audio_noise)

            combined.export(os.path.join(export_path, feature[:-3] + "wav"), format='wav')
            datasets.append({"combined_path": os.path.join(export_path[3:], feature[:-3] + "wav"),
                             "clean_path": os.path.join(feature_path[3:], feature)})

    return datasets


train_data = add_data(os.path.join(datasets_path, "clean_train"), os.path.join(datasets_path, "noise_train"),
                      os.path.join(datasets_path, "combined"))
test_data = add_data(os.path.join(datasets_path, "clean_test"), os.path.join(datasets_path, "noise_test"),
                     os.path.join(datasets_path, "combined"))

fields = ["combined_path","clean_path"]
train_filename = os.path.join(datasets_path, "train.csv")
test_filename = os.path.join(datasets_path, "test.csv")

with open(train_filename, "w", encoding="utf8", newline='') as train_csv:
    writer = csv.DictWriter(train_csv, fieldnames=fields)
    writer.writeheader()
    writer.writerows(train_data)

with open(test_filename, "w", encoding="utf8", newline='') as test_csv:
    writer = csv.DictWriter(test_csv, fieldnames=fields)
    writer.writeheader()
    writer.writerows(train_data)

