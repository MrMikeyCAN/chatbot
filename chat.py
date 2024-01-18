import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from utils import text_to_speech
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax

sound_path = "sound.mp3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jarvis"

class YourTextGenerationModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(YourTextGenerationModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

    def generate_text(self, start_sequence, max_length=50):
        generated_sequence = start_sequence.copy()

        for _ in range(max_length):
            input_sequence = torch.LongTensor([all_words.index(word) for word in generated_sequence if word in all_words])
            
            if len(input_sequence) > 0:  # Check if the sequence is not empty
                input_sequence_padded = pad_sequence([input_sequence], batch_first=True, padding_value=0)
                input_sequence_padded = input_sequence_padded.to(device)
                prediction = self.forward(input_sequence_padded)
                predicted_word_index = torch.argmax(prediction, dim=-1).item()
                predicted_word = all_words[predicted_word_index]
                generated_sequence.append(predicted_word)

                if predicted_word == "<EOS>":
                    break
            else:
                break  # Break if the input sequence is empty

        return " ".join(generated_sequence)

vocab_size = 10000
embedding_dim = 128
hidden_size = 256

text_gen_model = YourTextGenerationModel(vocab_size, embedding_dim, hidden_size).to(device)

while True:
    sentence = input("You: ")
    tokenized_sentence = tokenize(sentence)
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    if tag == "goodbye":
        text_to_speech(bot_name=bot_name, text="I am waiting for your orders sir!")
        break

    probs = softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.85:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                print(f"{bot_name}: {response}")
                text_to_speech(bot_name=bot_name,text=response)
    else:
        start_sequence = tokenized_sentence.copy()
        generated_response = text_gen_model.generate_text(start_sequence)
        print(f"{bot_name}: I don't understand, but here's something interesting: {generated_response}")
        text_to_speech(bot_name=bot_name,text="I don't understand, but here's something interesting: " + generated_response)
