# %%
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import datasets,  transforms
from torch.optim import lr_scheduler
import time
import copy
torch.cuda.empty_cache()
#IMPORT ALL MODEL ARCHITECTURE
from model_architectures import SimplifiedResNet, CNNModel, AttentionCNN
from preprocess import compute_mel_spectrogram
import cv2


#NOTE: WE HAVE IMPORTED BOTH THE MODEL ARCHITECTURES AND YOU CAN UNCOMMENT THE SECOND MODEL ARCHITECTURE INSTANTIATION TO TRAIN THAT MODEL
#NOTE: CURRENTLY, THIS FILE TRAINS THE FIRST MODEL ARCHITECTURE - SIMPLIFIED RESNET FOR 25 EPOCHS

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
                if audio_file == "Laughter_284.flac":
                  continue
                audio_path = os.path.join(class_dir, audio_file)
                self.data.append((audio_path, class_label))
                self.class_data[class_name] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, class_label = self.data[idx]
        #NOTE: THIS FUNCTION CALLS AUDIO PREPROCESS, COMPUTES SPECTROGRAM AND CONVERTS TO 3 CHANNEL
        mel_spec = compute_mel_spectrogram(audio_path)
        mel_spec = self.transform(mel_spec)
        return mel_spec, class_label - 1

# NO NEED FOR RESIZING AS MODELS DYNAMICALLY CALCULATE SIZE OF FULLY CONNECTED LAYERS BASED ON INPUT SIZE
# AND ALL SPECTROMGRAMS ARE OF SAME SIZE
transform = transforms.Compose([
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

# Define the directories [WHERE YOUR AUDIO FILES FOR TRAINING AND VALIDATING ARE STORED]
train_directory = "Project/train"
val_directory =  "Project/val"

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
def save_model(model, model_name):
  torch.save(model.state_dict(), f'{model_name}_weights.pth')
  torch.save(model, f'{model_name}.pth')

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        print("HELLO")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            index = 0
            print("STARTING ITERATION")
            for inputs, labels in dataloaders[phase]:
              print("BATCH NUMBER = ", index)
              # inputs = inputs.to(device)
              # labels = labels.to(device)

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

            if phase == 'train':
              print("STEPPING SCEHEDULER")
              scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'EPOCH: {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model weights for the model which has the highest acc.
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %%
# Define input shape and number of classes
input_shape = (128, 345, 3)
num_classes = 13  # Assuming 14 output classes

# Instantiate the model [UNCOMMENT IF WANT TO TRAIN ANOTHER MODEL]
model_ft = SimplifiedResNet(input_shape, num_classes)
# model_ft = AttentionCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)


#DIFFERENT SCHEDULER THIS TIME
exp_lr_scheduler =  lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%
model_ft = model_ft.to(device)

# [SimplifiedResNet] AUGMENTED [& ENSURED VAL HAS BEEN PREPROCESSED]
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

save_model(model_ft, "SimpleResNetStepSchedule")




