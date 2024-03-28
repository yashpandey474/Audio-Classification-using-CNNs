# %%
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
import optuna
from model_architectures import SimplifiedResNet, VGGish, AttentionCNN
torch.cuda.empty_cache()

# %%
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
            for audio_file in os.listdir(class_dir):
                audio_path = os.path.join(class_dir, audio_file)

                mel_spec = torch.zeros((128, 345, 3))
                
                #NOTE: THIS FUNCTION CALLS AUDIO PREPROCESS, COMPUTES SPECTROGRAM AND CONVERTS TO 3 CHANNEL
                try:
                    mel_spec = compute_mel_spectrogram(audio_path)
                    mel_spec = self.transform(mel_spec)
                    
                except:
                    print("ERROR FOR: ", audio_file)
                    continue

                self.data.append((mel_spec, class_label))
                self.class_data[class_name] += 1
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        mel_spec, class_label = self.data[idx]
        return mel_spec, class_label - 1
    
# %%
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # Optionally resize images
    transforms.ToTensor(),            # Convert images to tensors
])


# Define the mapping from class names to class indices
class_mapping = {
    'car_horn': 1,
    'dog_barking': 2,
    'drilling': 3,
    'Fart': 4,
    'Guitar': 5,
    'Gunshot_and_gunfire': 6,
    'Hi-hat': 7,
    'Knock': 8,
    'Laughter': 9,
    'Shatter': 10,
    'siren': 11,
    'Snare_drum': 12,
    'Splash_and_splatter': 13
}

# Define the directories
train_directory = "train"
val_directory = "val"

# Create datasets
train_dataset = MelSpecDataset(train_directory, class_mapping, transform)
val_dataset = MelSpecDataset(val_directory, class_mapping, transform)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


datasets = {"train": train_dataset, "val": val_dataset}
dataloaders = {"train": train_dataloader, "val": val_dataloader}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        print("HELLO")

        #NO VALIDATION PHASE AS THAT IS HAPPENING OUT OF TRAINING [ONLY FOR HYPERPARAMETER TUNING]
        for phase in ['train']:
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            index = 0
            print("STARTING ITERATION")
            for inputs, labels in dataloaders[phase]:
              print("BATCH NUMBER = ", index)

              index += 1
              optimizer.zero_grad()
              with torch.set_grad_enabled(phase == 'train'):
                  outputs = model(inputs)
                  _, preds = torch.max(outputs, 1)
                  loss = criterion(outputs, labels)

                  if phase == 'train':
                      loss.backward()
                      optimizer.step()

              running_loss += loss.item() * inputs.size(0)
              running_corrects += torch.sum(preds == labels.data)
                
            scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'EPOCH: {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return model

# %%

input_shape = (128, 345, 3)
num_classes = 13  # Assuming 14 output classes

def objective(trial):
    # Define the search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)

    # Instantiate the model
    model_ft = SimplifiedResNet(input_shape, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=gamma)
    
    # Train the model
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)
    
    # Evaluate on the validation set
    model_ft.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_dataloader:
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

# Perform hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Get the best hyperparameters
best_lr = study.best_params['lr']
best_gamma = study.best_params['gamma']
best_accuracy = study.best_value

print(f'Best LR: {best_lr}, Best Gamma: {best_gamma}, Best Validation Accuracy: {best_accuracy}')



