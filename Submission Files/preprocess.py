import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import datasets, models, transforms

# HYPERPARAMETERS
duration_seconds = 4
sample_rate = 44100
hyper_params = {
    'duration': duration_seconds*sample_rate,
     'n_mels': 128,
    'hop_length': 512,
    'n_fft': 2048,
    'fmin': 20,
    'fmax': sample_rate//2
}

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def apply_transform(mel_spec):
    return transform(mel_spec)

def audio_preprocess(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=44100)

    #normalising the waveform since each audio file has the amplitude values in different ranges
    waveform = waveform / np.max(np.abs(waveform))

    #keeping values greater than threshold
    waveform, index = librosa.effects.trim(waveform, top_db=60)

    # keeping values greater than threshold = 0.001
    wav = np.abs(waveform)
    mask = wav > 0.001     # 0.001 is equivalent to a 60db threshold
    waveform = waveform[mask]

    # pad to a length of 4s
    if len(waveform) > hyper_params['duration']:
        waveform = waveform[:hyper_params['duration']]
    else:
        padding = hyper_params['duration'] - len(waveform)
        offset = padding // 2
        waveform = np.pad(waveform, (offset, hyper_params['duration'] - len(waveform) - offset), 'constant')

    return waveform, sample_rate

def compute_mel_spectrogram(audio_file_path, n_fft=2048, hop_length=512, n_mels=128):
    # Load the audio file
    y, sr = audio_preprocess(audio_file_path)

    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.astype(np.float32)

    return mel_spec_db


def mono_to_color(X):
    # Convert single-channel image to three channels
    color_img = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)

    # Normalize pixel values to the range [0, 255]
    normalized_img = cv2.normalize(color_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_img