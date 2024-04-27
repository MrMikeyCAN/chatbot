source_texts = []

import pandas as pd

data = pd.read_csv("LLM.csv")

labels = data.context

for text in labels:
    source_texts.append(text)

with open("ModelData.txt", "r", encoding="utf-8") as file:
    for line in file:
        source_texts.append(line.strip().lower())

with open("ModelData.txt", "w", encoding="utf-8") as file:
    for text in source_texts:
        file.write(text + "\n")
