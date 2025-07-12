import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def load_audio(file_path):
    y, sr = librosa.load(file_path)
    print(f"Loaded: {file_path}")
    print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f}s, Sample Rate: {sr}")
    return y, sr

def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram (Log Scale)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_file = "UrbanSound8K/audio/fold1/101415-3-0-2.wav"  # Update this path
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"File not found: {audio_file}")

    y, sr = load_audio(audio_file)
    plot_waveform(y, sr)
    plot_spectrogram(y, sr)
