# %%
import os
import torch
import torchvision.transforms as transforms
import torchaudio.transforms as T
import torchvision.io as io
import torchaudio
import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import strip_silence
import random
drive.mount('/content/drive')

# %%
def augment_spectrograms(data_dir, output_dir, max_samples = 1000, time_mask_param = 80,
                         freq_mask_param = 80):
    # Iterate over the class distribution
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            # Count the number of files in the class folder
            file_names = [file_name for file_name in os.listdir(class_path)]
            num_files = len(file_names)

            print(f"Augmenting files in folder: {class_folder}")

            # Calculate the number of augmentations needed [ALTHOUGH EACH FILE WILL HAVE 800 SAMPLES AT THIS POINT]
            if num_files < max_samples:
                additional_files_needed = max_samples - num_files
                # Determine augmentation factor per file
                print(num_files, additional_files_needed)
                files_to_augment = random.sample(file_names, additional_files_needed)

            # count = 0
            # Loop through each file in the class folder
            for file_name in files_to_augment:
                if file_name == "Laughter_284.flac":
                  continue

                file_path = os.path.join(class_path, file_name)

                mel_spec = torch.from_numpy(np.load(file_path)['mel_spec'])

                augmented_spectogram = mel_spec.clone()

                time_masking = T.TimeMasking(time_mask_param=time_mask_param)
                freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

                # APPLY TIME MASKING
                augmented_spectogram = time_masking(augmented_spectogram)

                # APPLY FREQUENCY MASKING
                augmented_spectogram = freq_masking(augmented_spectogram)

                output_class_dir = os.path.join(output_dir, class_folder)
                os.makedirs(output_class_dir, exist_ok=True)

                save_mel_spec = augmented_spectogram.numpy()
                output_npz_path = os.path.join(output_class_dir, f"augmented_{file_name}")
                np.savez(output_npz_path, mel_spec=save_mel_spec)

            # print(f"CLASS: {class_folder} COUNT: {count}\n")

# %%
# Define directory paths
data_dir = '/content/drive/My Drive/DLproject-Numpy/train'
output_dir = '/content/drive/My Drive/DLproject-Numpy/augmented_spectograms'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

augment_spectrograms(data_dir, output_dir)


