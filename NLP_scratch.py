import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Uncomment these lines to download necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')

# Read the dataset
file = pd.read_csv("QA_dataset_combined.csv")

lemmatizer = WordNetLemmatizer()

# Tokenize and lemmatize the questions
X = file["Question"]
X = [nltk.word_tokenize(question) for question in X]
X = [[lemmatizer.lemmatize(word) for word in question] for question in X]

# Tokenize and lemmatize the answers
y = file["Answer"]
y = [nltk.word_tokenize(answer) for answer in y]
y = [[lemmatizer.lemmatize(word) for word in answer] for answer in y]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a custom dataset class
class QADataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        return question, answer


# Create instances of the dataset
train_dataset = QADataset(X_train, y_train)
val_dataset = QADataset(X_val, y_val)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Example to show the structure of the data loaders
for batch in training_loader:
    questions, answers = batch
    print("Questions:", questions)
    print("Answers:", answers)
    break
