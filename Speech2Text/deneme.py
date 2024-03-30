import sounddevice as sd
import numpy as np
import webrtcvad
import wave
import time
import noisereduce as nr

# Ses parametreleri
MAX = 80
DURATION = 20  # ms
RATE = 16000  # webrtcvad için uygun örnek hızı
CHANNELS = 1
CHUNK_SIZE = int(DURATION * RATE / 1000)  # 20 ms of audio

# VAD ayarı
vad = webrtcvad.Vad(2)  # Agresiflik seviyesi ayarı

# Ses akışını başlat
print("Kayıt başladı...")

silence_counter = 0
frames = []

def callback(indata, frames, time, status):
    global silence_counter

    # Gürültü azaltma
    audio = indata.flatten()
    reduced_noise = nr.reduce_noise(y=audio, sr=RATE)
    data = (reduced_noise * 32768.0).astype(np.int16)

    # VAD uygula
    is_speech = vad.is_speech(data.tobytes(), RATE)

    print(f"Ses var: {is_speech}", silence_counter)

    if not is_speech:
        silence_counter += 1
    else:
        silence_counter = 0

    if silence_counter < MAX:
        frames.extend(data)
    else:
        raise sd.CallbackStop

try:
    with sd.InputStream(samplerate=RATE, channels=1, dtype=np.int16, blocksize=CHUNK_SIZE, callback=callback):
        while silence_counter < MAX:
            sd.sleep(1000)  # Bekleme süresini bir saniye olarak ayarla
except KeyboardInterrupt:
    print("Kayıt durduruldu")
except sd.CallbackStop:
    pass

# Kaydı dosyaya yaz
file_name = "recording.wav"
wf = wave.open(file_name, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(2)  # int16 için 2 byte
wf.setframerate(RATE)
wf.writeframes(np.array(frames).tobytes())
wf.close()

print("Kayıt tamamlandı.")
