"""
Ben stt yapıp eğitene kadar geçici bir model elimizde hazır bulunsun diye benimki daha hızlı çalışacak bende çoklu
 dil yok ama tr için hemen hemen aynı performansı göstericek

"""
import time
from faster_whisper import WhisperModel
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

model_size = "medium"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = model.transcribe("cv-corpus-18.0-2024-06-14/tr/clips/common_voice_tr_17342455.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))