# Mikrofon alınınca tamamlanacak

import sounddevice as sd

def callback(indata, frames, time, status):
    if status:
        print(status)

    audio_data = indata[:, 0]

with sd.InputStream(samplerate=48000,blocksize=2048,
                    channels=1, callback=callback):
    sd.sleep(-1)