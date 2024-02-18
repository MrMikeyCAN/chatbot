import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(duration=5, fs=16000, filename='output.wav'):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Kaydın bitmesini bekleyin
    write(filename, fs, recording)
    print(f"Recording saved as {filename}")

# 5 saniyelik ses kaydını başlat
record_audio()

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf
import torch

def speech_to_text(audio_path, target_sampling_rate=16000):
    # Ses dosyasını yükle ve örnekleme oranını kontrol et
    speech, sampling_rate = sf.read(audio_path)
    
    # Eğer ses dosyasının örnekleme oranı hedeflenen örnekleme oranından farklıysa, yeniden örnekle
    if sampling_rate != target_sampling_rate:
        from scipy.signal import resample
        speech = resample(speech, int(target_sampling_rate / sampling_rate * len(speech)))
        sampling_rate = target_sampling_rate

    # Tokenizer ve modeli yükle
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Ses verisini modele uygun hale getir
    input_values = tokenizer(speech, return_tensors="pt", sampling_rate=sampling_rate).input_values

    # Tahmin yap
    logits = model(input_values).logits

    # Logitleri metne çevir
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])

    return transcription

transcription = speech_to_text("output.wav")
print(transcription)


