from gtts import gTTS
from pygame import mixer
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor


mixer.init()
mixer.music.set_volume(1)


def text_to_speech(bot_name: str, text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.mp3"
    tts.save(filename)
    print(f"{bot_name}: {text}")
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():
        continue

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

audio_input = [dataset[0]["audio"]["array"]]

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])

print(processed_dataset["input_values"][0].shape)