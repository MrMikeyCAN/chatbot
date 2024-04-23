## Mikrofon alınınca tamamlanacak

import sounddevice as sd
import vad

vad = vad.Vad()

def callback(indata, frames, time, status):
    if status:
        print(status)

    audio_data = indata[:, 0]
    print(vad.is_speech(audio_data, 48000)[0])

with sd.InputStream(samplerate=48000,blocksize=2048,
                    channels=1, callback=callback):
    sd.sleep(-1)