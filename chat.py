import torch
from torch.nn.utils.rnn import pad_sequence
from utils import tokenize, LanguageIndexMapper, text_to_speech
from model import ImprovedTransformerModel

# Load the trained model
FILE = "data.pth"
checkpoint = torch.load(FILE)

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]

model = ImprovedTransformerModel(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Load the language labels used during training
all_example_sentence = checkpoint["all_example_sentence"]
languages = checkpoint["languages"]

# Create an instance of LanguageIndexMapper
label_mapper = LanguageIndexMapper(languages)


# Function to predict the language of a sentence
def predict_language(sentence):
    model.eval()
    with torch.no_grad():
        # Tokenize the sentence
        words = tokenize(sentence)

        # Pad the sequence and convert to tensor
        words = pad_sequence([words], batch_first=True, padding_value=0).float()

        # Make prediction
        output = model(words)

        # Get the predicted label index
        _, predicted_index = torch.max(output, 1)

        # Map index to language label
        predicted_language = label_mapper.index_to_label_func(predicted_index.item())

        return predicted_language


# Example usage
new_sentence = "Merhabalar ben Mert, Vikipedia kullanıcısıyım Ayrıca teşekkürler."
predicted_language = predict_language(new_sentence)

text_to_speech(
    text=f"The predicted language for the sentence is: {predicted_language}",
    bot_name="Jarvis",
)
