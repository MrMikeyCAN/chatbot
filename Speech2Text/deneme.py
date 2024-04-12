from rnnoise_wrapper import RNNoise

denoiser = RNNoise()

audio = denoiser.read_wav('combined.wav')
denoised_audio = denoiser.filter(audio)
denoiser.write_wav('example.wav', denoised_audio)