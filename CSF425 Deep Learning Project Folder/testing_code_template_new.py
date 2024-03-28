# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
import torch
import librosa
import cv2
from torchvision import datasets, models, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TEST_DATA_DIRECTORY_ABSOLUTE_PATH = r"Project/val/Guitar"
OUTPUT_CSV_ABSOLUTE_PATH = "Project-Clone/output.csv"
# The above two variables will be changed during testing. The current values are an example of what their contents would look like.

import warnings

# Filter out all warnings
warnings.filterwarnings("ignore")

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
    transforms.ToTensor()
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

def mono_to_color(X):
    # Convert single-channel image to three channels
    color_img = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)

    # Normalize pixel values to the range [0, 255]
    normalized_img = cv2.normalize(color_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_img

def compute_mel_spectrogram(audio_file_path, n_fft=2048, hop_length=512, n_mels=128):
    # Load the audio file
    y, sr = audio_preprocess(audio_file_path)

    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.astype(np.float32)

    mel_spec_db = mono_to_color(mel_spec_db)
    
    return mel_spec_db



class SimplifiedResNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimplifiedResNet, self).__init__()
        self.in_channels = 32  # Reduced number of initial channels

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(ResNetBlockSimple, self.in_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(ResNetBlockSimple, self.layer1[0].conv2.out_channels, blocks=2, stride=2)
        self.layer3 = self._make_layer(ResNetBlockSimple, self.layer2[0].conv2.out_channels, blocks=2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(self.layer3[0].conv2.out_channels, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNetBlockSimple(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlockSimple, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    

class AttentionCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 43, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.attention = nn.Linear(128 * 16 * 43, 1)  # Attention layer

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output for attention mechanism
        x_flatten = x.view(x.size(0), -1)

        # Attention mechanism
        attention_weights = F.softmax(self.attention(x_flatten), dim=1)
        attention_output = torch.mul(x_flatten, attention_weights)

        # Fully connected layers
        x = F.relu(self.fc1(attention_output))
        x = self.fc2(x)
        return x
    
class VGGish(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGGish, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * (input_shape[0] // 16) * (input_shape[1] // 16), 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    

# The above two variables will be changed during testing. The current values are an example of what their contents would look like.
 # Instantiate the model
input_shape = (128, 345, 3)
num_classes = 13   

#NOTE: BOTH MODELS INSTANTIATED, UNCOMMENT THE EVALUATE WITH ARCHITECTURE 2 TO TEST THE SECOND MODEL
model_architecture1 = SimplifiedResNet(input_shape, num_classes)
model_architecture2 = VGGish(input_shape, num_classes)

# Define the file path for the saved model weights
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path relative to the current directory
model_weights1_path = os.path.join(current_dir, 'simplifiedResNet_25epoch_weights.pth')
model_weights2_path = os.path.join(current_dir, 'VGGish_90_weights.pth')

# Load the trained weights
model_architecture1.load_state_dict(torch.load(model_weights1_path))
model_architecture2.load_state_dict(torch.load(model_weights2_path))

# Set the model to evaluation mode
model_architecture1.eval()
model_architecture2.eval()

def evaluate(file_path, model):
    print("EVALUATING: ", file_path)

    # COMPUTE THE MEL SPECTOGRAM FOR THE AUDIO [FUNCTION PREPROCESSES AUDIO BEFORE CONVERSION]
    mel_spec = compute_mel_spectrogram(file_path)
    

    # ONLY FOR OUR TESTING [REMOVE BEFORE SUBMISSION]
    # mel_spec = np.load(file_path)['mel_spec']
    
    # APPLY TRANSFORMATIONS [CONVERTS TO TENSOR & OTHER TRANSFORMATIONS IF ANY]
    mel_spec = apply_transform(mel_spec).unsqueeze(0)

    # PREDICT THE CLASS
    with torch.no_grad():
        output = model(mel_spec)
        _, predicted = torch.max(output, 1) 

    return predicted.item() + 1



def test():
    filenames = []
    predictions = []
    # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # prediction = evaluate(file_path)
        absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        
        # prediction = evaluate(absolute_file_name, model_architecture1)

        #UNCOMMENT TO TEST THE SECOND MODEL ARCHITECTURE
        prediction = evaluate(absolute_file_name, model_architecture2)


        filenames.append(absolute_file_name)
        predictions.append(prediction)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)




# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
test()
# test_batch()
