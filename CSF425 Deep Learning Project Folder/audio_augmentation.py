# %%
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import strip_silence
import random
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image


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

input_folder = "/content/drive/My Drive/audio_dataset/audio_dataset/train"
output_folder = "/content/drive/My Drive/audio_dataset/audio_dataset/train"



def apply_audio_augmentation(audio, sample_rate, augmentation_type):
    # Load the audio file
    if augmentation_type == "time_shift":
        augmented_audio = np.roll(audio, 3000)
    elif augmentation_type == "speed_change":
        rate = random.uniform(0.7, 1.3)
        augmented_audio = librosa.effects.time_stretch(audio, rate=rate)
    elif augmentation_type == "pitch_shift":
        augmented_audio = librosa.effects.pitch_shift(audio, sr = sample_rate, n_steps=random.uniform(-2, 2))
    elif augmentation_type == "noise_injection":
        # Add white noise
        noise_factor = 0.005
        white_noise = np.random.randn(len(audio)) * noise_factor
        augmented_audio = audio + white_noise

    return augmented_audio

# %%
def augment_audio(input_path, output_path, max_samples=800):
    """
    Augments audio files in the input directory and saves them to the output directory.
    Only augments folders with less than max_samples files.
    """

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Loop through each class folder in the input directory
    for class_folder in os.listdir(input_path):
        class_path = os.path.join(input_path, class_folder)
        if os.path.isdir(class_path):
            # Count the number of files in the class folder
            file_paths = [file_name for file_name in os.listdir(class_path)]
            num_files = len(file_paths)

            print(f"Augmenting files in folder: {class_folder}")

            # Calculate the number of augmentations needed
            if num_files < max_samples:
                additional_files_needed = max_samples - num_files
                # Determine augmentation factor per file
                augmentation_factor_per_file = additional_files_needed // num_files
                # Calculate remaining augmentations
                remaining_augmentations = additional_files_needed % num_files
                
            else:
                augmentation_factor_per_file = 0
                remaining_augmentations = 0

            # Loop through each file in the class folder
            for file_name in file_paths:

                if file_name == "Laughter_284.flac":
                  continue

                # Load the audio file
                try:
                    audio_path = os.path.join(class_path, file_name)
                    waveform, sample_rate = librosa.load(audio_path, sr=44100)
                except:
                    continue

                # Determine the number of augmentations for this file
                augmentations_for_this_file = augmentation_factor_per_file
                if remaining_augmentations > 0:
                    augmentations_for_this_file += 1
                    remaining_augmentations -= 1

                # Apply augmentations
                for i in range(augmentations_for_this_file):
                    # Apply the chosen augmentation
                    augmentation_type = random.choice(["time_shift", "speed_change", "pitch_shift", "noise_injection"])
                    augmented_audio = apply_audio_augmentation(waveform, sample_rate, augmentation_type)

                    # Save the augmented audio to the output directory
                    output_file_name = f"aug_{i}_{os.path.splitext(file_name)[0]}.wav"
                    output_class_folder = os.path.join(output_path, class_folder)
                    os.makedirs(output_class_folder, exist_ok=True)
                    output_audio_path = os.path.join(output_class_folder, output_file_name)

                    sf.write(output_audio_path, augmented_audio, sample_rate)
                    print(f"Augmented file saved: {output_audio_path}")


            # print(class_folder, augmentation_factor_per_file, additional_files_needed, num_files, count)



#NOTE: UNCOMMENT THIS TO AUGMENT THE AUDIO
# augment_audio(input_folder, output_folder)


