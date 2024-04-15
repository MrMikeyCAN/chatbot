from nltk.tokenize import word_tokenize
import pandas as pd

data = pd.read_csv("ED.csv")
x = data["target"]
y = data["labels"]
max_number = 64


for i in range(len(x)):
    x[i] = word_tokenize(x[i])
    while(len(x[i])<max_number):
        x[i].append(" ")
    print(len(x[i]))

