import random
import json
import torch
from model import NeuralNet  # Use your NeuralNet class defined in model.py
from nltk_utils import tokenize, bag_of_words  # Use your functions from nltk_utils.py
from utils import text_to_speech

class Chatbot:
    def __init__(self, model, all_words, tags, intents):
        self.model = model
        self.all_words = all_words
        self.tags = tags
        self.intents = intents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.name = "Jarvis"

    def predict_class(self, sentence):
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        return tag, prob

    def get_response(self, tag):
        for intent in self.intents["intents"]:
            if tag == "goodbye":
                text_to_speech(bot_name=self.name, text="I am waiting for your orders sir!")
                return "Goodbye!"  # Add a return statement to exit the loop in the main code
            if tag == intent["tag"]:
                responses = intent.get("responses", [])  # Use get method to handle missing key
                if responses:
                    return text_to_speech(bot_name=self.name, text=random.choice(responses))
                else:
                    return text_to_speech(bot_name=self.name, text="I'm sorry, I don't have a response for that.")
        return text_to_speech(bot_name=self.name, text="I don't understand, can you ask something else?")

    def generate_text(self, prompt, max_length=50, temperature=1.0):
        prompt_tokens = tokenize(prompt)
        prompt_bow = bag_of_words(prompt_tokens, self.all_words).reshape(1, -1)
        prompt_bow = torch.tensor(prompt_bow, dtype=torch.float32).to(self.device)

        generated_words = []
        for _ in range(max_length):
            with torch.no_grad():
                output = self.model(prompt_bow)
            probabilities = torch.softmax(output / temperature, dim=1)
            predicted_word_idx = torch.multinomial(probabilities, 1).item()
            predicted_word = self.all_words[predicted_word_idx]
            generated_words.append(predicted_word)

            prompt += " " + predicted_word
            prompt_tokens = tokenize(prompt)
            prompt_bow = bag_of_words(prompt_tokens, self.all_words).reshape(1, -1)
            prompt_bow = torch.tensor(prompt_bow, dtype=torch.float32).to(self.device)

        return " ".join(generated_words)

    def chat(self, user_input):
        tag, prob = self.predict_class(user_input)
        if prob > 0.75:
            return self.get_response(tag)
        else:
            return text_to_speech(bot_name=self.name, text="I didn't understand.")

if __name__ == "__main__":
    data = torch.load("data.pth")
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(model_state)
    model.eval()

    intents = json.load(open("intents.json", "r"))
    chatbot = Chatbot(model, all_words, tags, intents)

    print("Chatbot'a hoş geldiniz! (çıkmak için 'çıkış' yazın)")
    while True:
        message = input("Sen: ")
        if message.lower() == "çıkış":  # Change "exit" to "çıkış"
            break
        response = chatbot.chat(message)
        generated_text = chatbot.generate_text(message, temperature=0.7)  # Adjust temperature if needed
        print(f"Chatbot: {response}")
        print(f"Generated Text: {generated_text}")
