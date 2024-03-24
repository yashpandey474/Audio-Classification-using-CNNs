import numpy as np
import os
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.nn.init as init
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa.display
import librosa
import zipfile
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import time
from PIL import Image
import copy
from collections import Counter
import cv2
torch.cuda.empty_cache()

def normalize_by_255(x):
    return x.float() / 255
    
class MelSpecDataset(Dataset):
    def __init__(self, directory, class_mapping, transform):
        self.directory = directory
        self.class_mapping = class_mapping
        self.data = []
        self.class_data = {}
        self.transform = transform

        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            self.class_data[class_name] = 0
            if not os.path.isdir(class_dir):
                continue
            class_label = self.class_mapping[class_name]  # Map class name to numerical label
            for npz_file in os.listdir(class_dir):
                npz_path = os.path.join(class_dir, npz_file)
                self.data.append((npz_path, class_label))
                self.class_data[class_name] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npz_path, class_label = self.data[idx]
        mel_spec = np.load(npz_path)['mel_spec']  # Assuming 'mel_spec' is the key for the mel spectrogram array
        mel_spec = self.transform(mel_spec)  # Apply the transform to the data
        return mel_spec, class_label - 1