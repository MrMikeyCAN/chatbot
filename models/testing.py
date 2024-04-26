source_texts = []

import pandas as pd

texts = pd.read_csv("train_essays_v1.csv")

texts_i_need = texts.iloc[:, :2].values
for text in texts_i_need:
    source_texts.append(text)

print(source_texts[:1])

# with open("input2.txt", "r", encoding="utf-8") as file:
#     for line in file:
#         source_texts.append(line.strip())

# with open("input2.txt", "w", encoding="utf-8") as file:
#     for text in source_texts:
#         file.write(text + "\n")
