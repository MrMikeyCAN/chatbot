import os
import shutil
from matplotlib import pyplot as plt
from Models import VAD_Model, STT_Model, Language_Detection
from Data_Filtering import Tensor_Converter
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import subprocess
import json

warnings.warn("Please Do Not Change Parameters Name In Json File")
# Close Warnings
warnings.filterwarnings("ignore")

"""
VAD
LD
STT
NF
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    with open("package.json", 'r', encoding='utf-8') as file:
        json_file = json.load(file)

    data_filtering_json_path = json_file["data_filtering_json_path"]
    data_filtering_py_path = json_file["data_filtering_py_path"]
    models_json_path = json_file["models_json_path"]
    data_filter = json_file["data_filter"]
    train_vad = json_file["train_vad"]
    train_ld = json_file["train_ld"]
    train_stt = json_file["train_stt"]
    draw_vad = json_file["draw_vad"]
    draw_ld = json_file["draw_ld"]
    draw_stt = json_file["draw_stt"]
    save_path = json_file["save_path"]
    save_on_it = json_file["save_on_it"]
    save_name = json_file["save_name"]
    count = json_file["save_count"]

    assert isinstance(data_filtering_json_path, str), "data_filtering_json_path should be a string"
    assert isinstance(data_filtering_py_path, str), "data_filtering_py_path should be a string"
    assert isinstance(models_json_path, str), "models_json_path should be a string"
    assert isinstance(data_filter, bool), "data_filter is a boolean"
    assert isinstance(train_vad, bool), "train_vad should be a boolean"
    assert isinstance(train_ld, bool), "train_ld should be a boolean"
    assert isinstance(train_stt, bool), "train_stt should be a boolean"
    assert isinstance(draw_vad, bool), "draw_vad should be a boolean"
    assert isinstance(draw_ld, bool), "draw_ld should be a boolean"
    assert isinstance(draw_stt, bool), "draw_stt should be a boolean"
    assert isinstance(save_path, str), "save_path should be a string"
    assert isinstance(save_on_it, bool), "save_on_it should be a boolean"
    assert isinstance(save_name, str), "save_name should be a string"
    assert isinstance(count, int), "count should be an integer"

    save_path_files = os.listdir(save_path)
    for save_path_file in save_path_files:
        os.remove(os.path.join(save_path, save_path_file))

    with open(models_json_path, 'r', encoding='utf-8') as file:
        model_json_file = json.load(file)

    json_model_names = ["VAD", "LD", "STT"]

    count = max(0, count)

    vad = json_file[json_model_names[0]]
    ld = json_file[json_model_names[1]]
    stt = json_file[json_model_names[2]]

    for model_name in json_model_names:
        json_model = json_file[model_name]
        assert isinstance(json_model, list), f"{model_name} should be a list"
        assert isinstance(json_model[0], float), f"{model_name} sample_fraction should be a float."
        assert isinstance(json_model[1], int), f"{model_name} batch_size should be an int."
        assert isinstance(json_model[2], int), f"{model_name} epoch should be an int."
        assert isinstance(json_model[3], float), f"{model_name} lr should be an float."
        assert isinstance(json_model[4], float), f"{model_name} weight_decay should be an float."
        assert isinstance(json_model[5], int), f"{model_name} frequency should be an int."

        assert 0.001 <= json_model[0] <= 1, f"{model_name} sample_fraction should be between 0 and 1."
        assert 0 < json_model[1], f"{model_name} batch_size should be greater than 0."

except KeyError:
    raise KeyError("Please Control Json File.")

except FileNotFoundError:
    raise FileNotFoundError("Json File Not Found.")

if data_filter:
    subprocess.run(["python", data_filtering_py_path])


def train_function(num_epochs, train_dataloader, val_dataloader, model, criterion, optimizer, frequency,
                   process, threshold=0, alphabet=None):

    global save_path, save_on_it, save_name, count
    loss_history = list()
    accuracy_history = list()
    iteration_history = list()
    count = 0
    for epoch in range(num_epochs):
        for audio_train, label_train in train_dataloader:
            optimizer.zero_grad()

            new_audio_train = audio_train.permute(0, 2, 1).to(device)
            new_label_train = label_train.to(device)

            if process is not "STT":
                pred = model(new_audio_train)
                if process == "VAD":
                    pred = VAD_Model.sigmoid(pred)
            else:
                pred = model(new_audio_train, new_label_train)

            loss = criterion(pred, new_label_train)
            loss.backward()

            optimizer.step()

            count += 1

            if count % frequency == 0:
                correct_number = 0
                total = 0
                for audio_val, label_val in val_dataloader:
                    new_audio_val = audio_val.permute(0, 2, 1).to(device)
                    new_label_val = label_val.to(device)

                    if process == "VAD":
                        pred = model(new_audio_val)
                        pred = VAD_Model.sigmoid(pred)
                        pred = torch.where(pred >= threshold, torch.tensor(1), torch.tensor(0))
                    elif process == "LD":
                        pred = model(new_audio_val)
                        pred = pred.argmax(dim=1)
                        new_label_val = label_val.argmax(dim=1)
                    elif process == "STT":
                        pred = model(new_audio_val, new_label_val)
                        pred = pred.argmax(dim=1)
                        print(new_label_val.shape)

                    else:
                        raise ValueError("Process Type Error")

                    if process is not "STT":
                        total += len(new_label_val)
                        correct_number += sum([p == l for p, l in zip(pred, new_label_val)])
                    else:
                        total += len(new_label_val) * new_label_val.size(1)
                        for p, l in zip(pred, new_label_val):
                            correct_number += sum([p_b == l_b for p_b, l_b in zip(p, l)])

                accuracy = 100 * correct_number / float(total)

                loss_history.append(loss.item())
                accuracy_history.append(accuracy)
                iteration_history.append(count)

                print(f"Epochs [{epoch}/{num_epochs}] Iteration {count} Loss {loss.item():.4f} Accuracy {accuracy:.4f}")

                if process is "STT":
                    for i in pred:
                        prediction = ""
                        for j in i:
                            prediction += alphabet[j - 2]
                        print(prediction)

                saving = os.path.join(save_path, save_name)
                if save_on_it:
                    torch.save(model.state_dict(), str(saving))
                else:
                    while os.path.exists(saving):
                        save_name = save_name[:save_name.rfind('.')] + str(count) + save_name[save_name.rfind('.'):]
                        saving = os.path.join(save_path, save_name)
                        count += 1
                    torch.save(model.state_dict(), str(saving))

    return iteration_history, loss_history, accuracy_history


def pre_data(data_parameters, process):
    converter = Tensor_Converter.TensorConverter(sample_fraction=data_parameters[0], batch_size=data_parameters[1], process=process,
                                                 json_filename=data_filtering_json_path)
    train_dataloader, val_dataloader, test_dataloader = converter.process_data()

    # Model Parameters
    parameters = model_json_file[process]
    assert isinstance(parameters, dict), "VAD should be a dict"

    input_size = converter.input_size
    hidden_size = parameters["hidden_size"]
    num_layers = parameters["num_layers"]

    if process is not "STT":
        num_classes = parameters["num_classes"]
        assert isinstance(num_classes, int), "num_classes should be an int."
        num_classes = max(1, num_classes)

    dropout = parameters["dropout"]

    threshold = 0

    if process == "VAD":
        threshold = parameters["threshold"]
        assert isinstance(threshold, float), "vad_threshold should be a float."
        assert 0 <= threshold <= 1, "vad_threshold should be between 0 and 1."
        assert 0 <= dropout <= 1, "dropout should be between 0 and 1."
    elif process == "STT":
        max_length = parameters["max_length"]
        assert isinstance(max_length, int), "max_length should be an int."
        max_length = max(1, max_length)

    assert isinstance(hidden_size, int), "hidden_size should be an int."
    assert isinstance(num_layers, int), "num_layers should be an int."
    assert isinstance(dropout, float), "dropout should be a float."

    hidden_size = max(1, hidden_size)
    num_layers = max(1, num_layers)

    if process == "VAD":
        model = VAD_Model.VAD(input_size, hidden_size, num_classes, num_layers, dropout).to(device)
        criterion = nn.BCELoss()
    elif process == "LD":
        model = Language_Detection.LD(input_size, hidden_size, num_classes, num_layers, dropout).to(device)
        criterion = nn.CrossEntropyLoss()
    elif process == "STT":
        model = STT_Model.STT(input_size, hidden_size, len(converter.alphabet)+2, num_layers, dropout,
                              max_length).to(device)
        criterion = nn.NLLLoss()
    else:
        raise ValueError("Process should be either VAD or LD.")

    num_epochs = max(data_parameters[2], 1)
    learning_rate = data_parameters[3]
    weight_decay = data_parameters[4]
    frequency = max(data_parameters[5], 1)

    assert learning_rate >= 0, "learning_rate should be a positive number."
    assert weight_decay >= 0, "weight_decay should be a positive number."

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return train_function(num_epochs, train_dataloader, val_dataloader,
                          model, criterion, optimizer, frequency, process, threshold=threshold,
                          alphabet=converter.alphabet)


def draw_loss_accuracy(iteration_list, loss_list, accuracy_list):
    plt.plot(iteration_list, loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of iteration")
    plt.show()

    plt.plot(iteration_list, accuracy_list, color="red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of iteration")
    plt.show()


# VAD
if train_vad:
    history = pre_data(vad, "VAD")
    if draw_vad:
        draw_loss_accuracy(history[0], history[1], history[2])

# LD
if train_ld:
    history = pre_data(ld, "LD")
    if draw_ld:
        draw_loss_accuracy(history[0], history[1], history[2])

# STT
if stt:
    history = pre_data(stt, "STT")
    if draw_stt:
        draw_loss_accuracy(history[0], history[1], history[2])