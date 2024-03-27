# %%
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import cv2

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

# %%
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

# %%
def save_mel_spectrogram_as_image(mel_spec, output_image_path):
    # Plot the mel spectrogram without axes and color bar
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, hop_length=512, x_axis=None, y_axis=None)
    plt.axis('off')  # Turn off axes

    # Save the mel spectrogram as an image without padding
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# %%
def process_label_folder(label_directory, output_label_directory):
    # Loop through each audio file in the label directory
    for audio_file in os.listdir(label_directory):
        audio_file_path = os.path.join(label_directory, audio_file)

        try:
            # Compute mel spectrogram
            mel_spec = compute_mel_spectrogram(audio_file_path)

            # Convert to 3 channels [Better for inputting to CNN model]
            mel_spec = mono_to_color(mel_spec)


            output_npz_path = os.path.join(output_label_directory, os.path.splitext(audio_file)[0] + '.npz')
            np.savez(output_npz_path, mel_spec=mel_spec)

            print(f'Saved mel spectrogram array: {output_npz_path}')
        except Exception as e:
            print(f'Error processing {audio_file}: {str(e)}')
            continue



# %%
def process_audio_dataset(root_directory, output_directory):
    # Loop through 'train' and 'val' subfolders
    for subfolder in ["train", "val"]:
        subfolder_directory = os.path.join(root_directory, subfolder)

        # Loop through each label folder in the 'train' or 'val' subfolder
        for label in os.listdir(subfolder_directory):
        # for label in ['Laughter', 'Shatter', 'Snare_drum', 'Splash_and_splatter', 'siren']:
            label_directory = os.path.join(subfolder_directory, label)

            # Create a corresponding output subfolder for the label
            output_label_directory = os.path.join(output_directory, subfolder, label)
            os.makedirs(output_label_directory, exist_ok=True)

            # Process audio files in label folder
            process_label_folder(label_directory, output_label_directory)


# %%
def main():
    # Define the root directory containing 'audio_dataset'
    root_directory = '/content/drive/My Drive/audio_dataset/audio_dataset'
    # root_directory = '/content/drive/My Drive/DLproject-Numpy'

    # Define the directory where mel spectrogram images will be saved
    output_directory = '/content/drive/My Drive/DLproject-Numpy'

    # Process the audio dataset
    process_audio_dataset(root_directory, output_directory)



if __name__ == "__main__":
    # NOTE: UNCOMMENT TO CREATE SPECTROGRAMS FROM AUDIOS IN ROOT DIRECTORY TO OUTPUT DIRECTORY
    # main()
    pass

