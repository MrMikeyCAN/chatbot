from datasets import load_dataset

dataset = load_dataset("covost2", "tr_en",data_dir="tr")
print(dataset["train"][0])