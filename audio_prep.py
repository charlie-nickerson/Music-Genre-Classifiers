import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


file = "blues.00001.wav"

# Show wave form
signal, sr = librosa.load(file, sr=220650) # sr is sample rate, signal = sr*time -> 22050 * 30
# librosa.display.waveshow(signal, sr=sr)    # If you are not running librosa 0.9.0 use waveplot instead
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

#fft -> spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft) # Magnitude indicates the contribution each frequency adds to the overall sound
frequency = np.linspace(0, sr, len(magnitude))

# Cut out the redundant part of the spectragram
left_freq = frequency[:int(len(frequency)/2)]
left_mag = magnitude[:int(len(frequency)/2)]

# plt.plot(left_freq, left_mag)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

#stft -> spectrogram
n_fft = 2048 # Express as number of samples
hop_length = 512 # Amount to shift the fourier transfer to ther right
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

spectrogram = np.abs(stft)
#converts spectrogram to a logarithmic scale
log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

#MFCCs
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13) # n_mfcc is the number of coefficients
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
