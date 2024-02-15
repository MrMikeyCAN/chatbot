# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
# import torch
# from datasets import load_dataset
# from transformers import SpeechT5HifiGan
# import soundfile as sf  # Library for writing audio files
# import soundfile as sd  # Library for playing audio


### * Çalışan kodu class haline getirip çağırılabilir hale çevirdim
# class Speech_to_text:
#     def __init__(self):
#         self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
#         self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
#         self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

#         # Load embeddings dataset
#         self.embeddings_dataset = load_dataset(
#             "Matthijs/cmu-arctic-xvectors", split="validation"
#         )
#         self.speaker_embeddings = torch.tensor(
#             self.embeddings_dataset[7306]["xvector"]
#         ).unsqueeze(0)

#     def generate_speech(self, text):
#         # Generate speech using T5 model
#         inputs = self.processor(text=text, return_tensors="pt")
#         speech = self.model.generate_speech(
#             inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
#         )

#         # Check if speech is generated
#         if speech is not None:
#             print("Speech generated successfully.")

#             # Save speech to a file
#             output_audio_file = "output_audio.wav"
#             sf.write(output_audio_file, speech, 16000)

#             print(f"Speech saved to {output_audio_file}.")

#             return output_audio_file
#         else:
#             print("No speech generated.")
#             return None

#     def play_audio(self, audio_file):
#         # Play audio
#         data, fs = sf.read(audio_file, dtype="float32")
#         sd.play(data, fs)
#         status = sd.wait()  # Wait until playback is finished


# # TODO örnek kullanım:
# speech_generator = Speech_to_text()
# audio_file = speech_generator.generate_speech("Hello sir thanks for your help")
# if audio_file:
#     speech_generator.play_audio(audio_file)


# # TODO Bark model ile düzenlenen ai
# #! IPyhton ses verme konusunda sorunları var

# from transformers import BarkModel, BarkProcessor

# model = BarkModel.from_pretrained("suno/bark-small")
# processor = BarkProcessor.from_pretrained("suno/bark-small")

# # * english version
# inputs = processor(
#     text="Hello sir",
#     voice_preset="v2/en_speaker_6",
# )

# speech_output_en = model.generate(**inputs).cpu().numpy().squeeze()

# print(speech_output_en)

# from scipy.io.wavfile import write as write_wav

# # save audio to disk, but first take the sample rate from the model config
# sample_rate = model.generation_config.sample_rate
# write_wav("bark_generation_en.wav", sample_rate, speech_output_en)


# from transformers import pipeline

# pipe = pipeline("text-to-speech", model="suno/bark-small")
# text = "[clears throat] This is a test ... and I just took a long pause."
# output = pipe(text)


# from IPython.display import Audio
# from scipy.io.wavfile import write as write_wav
# Audio(output["audio"], rate=output["sampling_rate"])


### **** Dataset harici kendimiz özelleştirebileceğimiz kodlar **** ###
# TODO connecting hugging face to import the dataset
from huggingface_hub import notebook_login

### ? if you didn't logged in even once change the new_session to True
notebook_login(
    write_permission=True,
)


from datasets import load_dataset, Audio

### TODO importing the dataset # ! its important to say trust_remote_code or it is not going to work
### ? TODO We will change this dataset to our own needs

dataset = load_dataset(
    "facebook/voxpopuli", "nl", split="train", trust_remote_code=True
)

### * checking for is dataset equal to website, if it has 20986 its true
print(len(dataset))


### * dataset has sampling rate of 16khz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


### TODO Preprocecssing the data

from transformers import SpeechT5Processor

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)


### TODO Text cleanup
tokenizer = processor.tokenizer


def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

### ? unfortuneally for this example of usage there is no letters like ö and ü
# ? print(dataset_vocab - tokenizer_vocab) = {'ö', 'ç', 'è', 'ü', 'ë', 'ï', ' ', 'à', 'í'}

### TODO Cleaning up the dataset as it has to

replacements = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]


def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs


dataset = dataset.map(cleanup_text)


### TODO creating the speaker

from collections import defaultdict

speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1


### TODO matplotlib for picturezaing the examples
### ? I commented it for coding faster
# import matplotlib.pyplot as plt

# plt.figure()
# plt.hist(speaker_counts.values(), bins=20)
# plt.ylabel("Speakers")
# plt.xlabel("Examples")
# plt.show()


### TODO filtering the speakers for own usage
### ! Personalized usage is important
def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400


dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

### * print(len(set(dataset["speaker_id"])))
### ? len of speaker is for me 42 and also in website but we have to personalize it later


### TODO looking for dataset len after filtering
# ? print(len(dataset))
# ? for this dataset it has to be 9973


import os
import torch
from speechbrain.pretrained import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings


### TODO speaker embeddings for training data much more
### ? For no its copied-code for more able to learn

import os
import torch
from speechbrain.pretrained import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings


### TODO preprocessing the data once more because of embedding


def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example


### TODO checking for any type of error

processed_example = prepare_dataset(dataset[0])
print(list(processed_example.keys()))
# * Result = ['input_ids', 'labels', 'speaker_embeddings']


print(processed_example["speaker_embeddings"].shape)

### TODO visualating the data for checking

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(processed_example["labels"].T)
# plt.show()


### TODO applying preprocessing for the entire dataset
### ! it will take a time about 5-10

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)


def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
### TODO checking for okay or not. For this dataset it has to be 8259
# print(len(dataset))


### TODO making basic train-test split
dataset = dataset.train_test_split(test_size=0.1)

### TODO collating datas for adding padding for the longest data of the dataset
### ? This is also copied-code but its doesnt matter for other datasets

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch


data_collator = TTSDataCollatorWithPadding(processor=processor)


### TODO Training the model

from transformers import SpeechT5ForTextToSpeech

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)


### ! normally it has to be True but for training its important to be False
model.config.use_cache = False


### TODO Defining the training arguments
### ? It will personalized for our own dataset later

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="text_to_speech_pretrained_model",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    ### ? Change this code to True if you have cuda
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)


### TODO Training the dataset with training arguments

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)


trainer.train()

### ? saving model but change the name of yours or any string you want with "KMCan" and dont forget to add it on gitignore like /{your_name}/
processor.save_pretrained("KMCan/text_to_speech_pretrained_model")


### TODO pushing the final result to hub
# ! Its important for later.While checking for our dataset
trainer.push_to_hub(commit_message="Trying to figure it out")


### TODO Using the pretrained data with two optional way

## TODO 1. way of using Pipelines

from transformers import pipeline

pipe = pipeline("text-to-speech", model="KMCan/text_to_speech_pretrained_model")


text = "Hello sir"


example = dataset["test"][304]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)


forward_params = {"speaker_embeddings": speaker_embeddings}
output = pipe(text, forward_params=forward_params)

### ? Checking for the output as always
# print(output)

### TODO listening the audio
### ? I am using soundfile but if you want you can use any library you want

import soundfile as sf

# Assuming output['audio'] contains raw audio data and output['sampling_rate'] contains the sampling rate
audio_data = output["audio"]
sampling_rate = output["sampling_rate"]

import soundfile as sf

# Assuming output['audio'] contains raw audio data and output['sampling_rate'] contains the sampling rate
audio_data = output["audio"]
sampling_rate = output["sampling_rate"]

# Save the audio data to a temporary file
temp_file = "temp_audio.wav"
sf.write(temp_file, audio_data, sampling_rate)
