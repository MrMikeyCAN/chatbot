import random
import json
import torch
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from utils import text_to_spech
from pygame import mixer

### Ses çalma ayarları
sound_path = "sound.waw"
mixer.init()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r+") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
conversation_history = intents


def process_command(command):
    try:
        result = exec(command)
        return result
    except Exception as e:
        print(f"Error executing command: {e}")


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jarvis"
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")

    tokenized_sentence = tokenize(sentence)
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    if tag == "goodbye":
        text_to_spech(text="I am waiting for your orders sir!", bot_name=bot_name)
        mixer.music.load(sound_path)
        mixer.music.play()
        while mixer.music.get_busy():
            continue
        break

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.65:
        for intent in conversation_history["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                intent["patterns"].append(sentence)
                intent["responses"].append(response)
                text_to_spech(bot_name=bot_name, text=response)
                mixer.music.load(sound_path)
                mixer.music.play()
                while mixer.music.get_busy():
                    continue

    else:
        print(f"{bot_name}: I do not understand...")
