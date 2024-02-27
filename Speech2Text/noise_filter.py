from pedalboard.io import AudioFile
from pedalboard import *
import noisereduce as nr
import sounddevice as sd

class SoundFilter:
    def __init__(self,sample_rate=16000):
        self.sample_rate=sample_rate

    def read(self,filename):
        with AudioFile(filename).resampled_to(self.sample_rate) as f:
            self.audio = f.read(f.frames)

    def clear(self,audio=None,stationary=True,prop_decrease=0.75,
        board:Pedalboard = None,time=3):

        if(self.audio is None and audio is not None):
            self.audio=audio
        
        if(time == 0):
            reduced_noise = nr.reduce_noise(y=self.audio, sr=self.sample_rate, stationary=stationary, prop_decrease=prop_decrease)
        else:
            noise_clip = self.audio[0:int(time * self.sample_rate)]
            reduced_noise = nr.reduce_noise(y=self.audio, sr=self.sample_rate,y_noise=noise_clip)

        if board is not None:
            self.effected = board(reduced_noise, self.sample_rate)
        else:
            self.effected = reduced_noise

    def write(self,filename):
        if(self.effected is None):
            print("No effect")
        else:
            with AudioFile(filename, 'w', self.sample_rate, self.effected.shape[0]) as f:
                f.write(self.effected)

    def playSounds(self):
        print('original audio')
        sd.play(self.audio[0],self.sample_rate)
        sd.wait()

        print('enhanced audio')
        sd.play(self.effected[0],self.sample_rate)
        sd.wait()
