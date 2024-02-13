from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import torch
from datasets import load_dataset
from transformers import SpeechT5HifiGan
import soundfile as sf  # Library for writing audio files
import sounddevice as sd  # Library for playing audio


### * Çalışan kodu class haline getirip çağırılabilir hale çevirdim
class Speech_to_text:
    def __init__(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load embeddings dataset
        self.embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_embeddings = torch.tensor(
            self.embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0)

    def generate_speech(self, text):
        # Generate speech using T5 model
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(
            inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
        )

        # Check if speech is generated
        if speech is not None:
            print("Speech generated successfully.")

            # Save speech to a file
            output_audio_file = "output_audio.wav"
            sf.write(output_audio_file, speech, 16000)

            print(f"Speech saved to {output_audio_file}.")

            return output_audio_file
        else:
            print("No speech generated.")
            return None

    def play_audio(self, audio_file):
        # Play audio
        data, fs = sf.read(audio_file, dtype="float32")
        sd.play(data, fs)
        status = sd.wait()  # Wait until playback is finished


# TODO örnek kullanım:
speech_generator = Speech_to_text()
audio_file = speech_generator.generate_speech("Hello sir thanks for your help")
if audio_file:
    speech_generator.play_audio(audio_file)


# TODO Bark model ile düzenlenen ai
#! IPyhton ses verme konusunda sorunları var

# from transformers import BarkModel, BarkProcessor

# model = BarkModel.from_pretrained("suno/bark-small")
# processor = BarkProcessor.from_pretrained("suno/bark-small")

# # * english version
# inputs = processor(
#     "This is a test!", voice_preset="v2/en_speaker_3", return_attention_mask=True
# )

# speech_output = model.generate(**inputs).cpu().numpy()

# # * french version
# inputs = processor(
#     "C'est un test!", voice_preset="v2/fr_speaker_1", return_attention_mask=True
# )

# speech_output = model.generate(**inputs).cpu().numpy()


# inputs = processor(
#     "[clears throat] This is a test ... and I just took a long pause.",
#     voice_preset="v2/fr_speaker_1",
#     return_attention_mask=True,
# )

# speech_output = model.generate(**inputs).cpu().numpy()


# inputs = processor(
#     "♪ In the mighty jungle, I'm trying to generate barks.", return_attention_mask=True
# )

# speech_output = model.generate(**inputs).cpu().numpy()


# input_list = [
#     "[clears throat] Hello uh ..., my dog is cute [laughter]",
#     "Let's try generating speech, with Bark, a text-to-speech model",
#     "♪ In the jungle, the mighty jungle, the lion barks tonight ♪",
# ]

# # also add a speaker embedding
# inputs = processor(
#     input_list, voice_preset="v2/en_speaker_3", return_attention_mask=True
# )

# speech_output = model.generate(**inputs).cpu().numpy()

# from IPython.display import Audio

# sampling_rate = model.generation_config.sample_rate
# Audio(speech_output[0], rate=sampling_rate)


# Audio(speech_output[1], rate=sampling_rate)

# Audio(speech_output[2], rate=sampling_rate)
