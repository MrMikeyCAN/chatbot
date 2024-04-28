source_texts = []

end_token = "<END>"


print(source_texts[:1])

with open("input2.txt", "r", encoding="utf-8") as f:
    data = f.read().lower().strip().split("\n")

for text in data:
    lines_with_end = text + " " + end_token + " "
    source_texts.append(lines_with_end)

with open("input2.txt", "w", encoding="utf-8") as file:
    for text in source_texts:
        file.write(text + "\n")
