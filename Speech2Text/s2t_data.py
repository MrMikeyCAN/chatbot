## Yeni başlandı stt

from datasets import load_dataset
import time

start = time.time()

dataset_tr = load_dataset("covost2", "tr_en", data_dir="Datasets/tr", trust_remote_code=True)
# dataset_zh_CH = load_dataset("covost2", "zh-CN_en",data_dir="Datasets/zh-CN", trust_remote_code=True)

end = time.time()
print(end-start, "second")

