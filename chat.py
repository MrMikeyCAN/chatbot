import torch
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from utils import text_to_speech


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

    def generate_text(self, start_sequence, max_length=50, all_words=None, device=None):
        generated_sequence = start_sequence.clone()

        for _ in range(max_length):
            input_sequence = torch.LongTensor(
                [
                    start_sequence.tolist().index(word)
                    for word in generated_sequence
                    if word in start_sequence.tolist()
                ]
            )

            if len(input_sequence) > 0:  # Check if the sequence is not empty
                input_sequence_padded = pad_sequence(
                    [input_sequence], batch_first=True, padding_value=0
                )
                input_sequence_padded = input_sequence_padded.to(device)
                prediction = self.forward(input_sequence_padded)
                predicted_word_index = torch.argmax(prediction, dim=-1).item()
                predicted_word = all_words[predicted_word_index]
                generated_sequence.append(predicted_word)

                if predicted_word == "<EOS>":
                    break
            else:
                break  # Break if the input sequence is empty

        generated_sequence = [str(word) for word in generated_sequence if str(word) != '0.0']
        
        return " ".join(generated_sequence)


class Chatbot:
    def __init__(self, config):
        cuda_available = torch.cuda.is_available() and config.get("device", "cuda")
        self.device = torch.device("cuda" if cuda_available else "cpu")
        self.all_words = None
        self.tags = None
        self.intents = None  # Add this line
        self.load_model(config["model_file"])
        self.bot_name = config.get("bot_name", "Jarvis")

    def load_model(self, model_file):
        data = torch.load(model_file)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        self.intents = data.get("intents", {})  # Handle missing 'intents' key

        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.text_gen_model = YourTextGenerationModel(
            config.get("vocab_size", 10000),
            config.get("embedding_dim", 128),
            config.get("hidden_size", 256),
        ).to(self.device)

    def process_input(self, user_input):
        tokenized_sentence = tokenize(user_input)
        X = bag_of_words(tokenized_sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        return torch.from_numpy(X).to(self.device)

    def generate_response(self, input_tensor, start_sequence, input_text):
        output = self.model(input_tensor)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        if tag == "goodbye":
            text_to_speech(
                bot_name=self.bot_name, text="I am waiting for your orders sir!"
            )
            exit()

        probs = softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        print(f"Tag: {tag}, Probability: {prob.item()}")

        if prob.item() > 0.85:
            for intent in self.intents.get("intents", []):
                if tag == intent.get("tag"):
                    generated_response = self.text_gen_model.generate_text(
                        start_sequence, all_words=self.all_words, device=self.device
                    )
                    print(f"Generated Response: {generated_response}")
                    text_to_speech(bot_name=self.bot_name, text=generated_response)
        else:
            start_sequence = self.process_input(input_text)  # Use the processed input
            generated_response = self.text_gen_model.generate_text(
                start_sequence, all_words=self.all_words, device=self.device
            )
            print(f"Generated Response: {generated_response}")
            text_to_speech(bot_name=self.bot_name, text=generated_response)


    def run(self):
        start_sequence = []
        while True:
            user_input = input("You: ")
            input_tensor = self.process_input(user_input)
            self.generate_response(input_tensor, start_sequence, user_input)


if __name__ == "__main__":
    config = {
        "model_file": "data.pth",
        "vocab_size": 10000,
        "embedding_dim": 128,
        "hidden_size": 256,
        "device": "cuda",  # Adjust as needed
    }
    chatbot = Chatbot(config)
    chatbot.run()
