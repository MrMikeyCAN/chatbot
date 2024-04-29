# Yeni başlandı stt

from datasets import load_dataset

dataset_tr = load_dataset("covost2", "tr_en", data_dir="../Datasets/STT_Datasets/tr", trust_remote_code=True)
# dataset_zh_CH = load_dataset("covost2", "zh-CN_en",data_dir="Datasets/zh-CN", trust_remote_code=True)
print(dataset_tr["train"][0])




