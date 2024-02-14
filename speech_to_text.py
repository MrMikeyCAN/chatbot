from transformers import pipeline

# Ses dosyasının yolu
audio_file_path = "sound.wav"

# Ses dosyasından metne dönüştürme modelini yükle
speech_to_text = pipeline("automatic-speech-recognition")

# Ses dosyasını metne dönüştür
transcript = speech_to_text(audio_file_path)

# Elde edilen metni yazdır
print(transcript[0]["transcription"])
