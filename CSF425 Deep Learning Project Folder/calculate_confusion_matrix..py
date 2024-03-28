import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from model_architectures import SimplifiedResNet, AttentionCNN, CNNModel, VGGish
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
#IMPORT ALL MODEL ARCHITECTURE
from model_architectures import SimplifiedResNet, CNNModel, AttentionCNN
from preprocess import compute_mel_spectrogram
import librosa 
import numpy as np

input_shape = (128, 345, 3)
num_classes = 13 

model_ResNet = SimplifiedResNet(input_shape, num_classes)
model_Attention = AttentionCNN(num_classes)
model_CNN = CNNModel(input_shape, num_classes)
model_VGGish = VGGish(input_shape, num_classes)

# Load the model weights
model_ResNet.load_state_dict(torch.load("Project-Clone/Submission Files/simplifiedResNet_25epoch_weights.pth"))
model_Attention.load_state_dict(torch.load("Project-Clone/Submission Files/AttentionCNN.pth"))
model_CNN.load_state_dict(torch.load("Project-Clone/Model/CNNModel_10epochs_weights.pth"))
model_VGGish.load_state_dict(torch.load("Project-Clone/Submission Files/VGGish_90_weights.pth"))

models = {
    "ResNet": model_ResNet,
    "Attention": model_Attention,
    "CNN": model_CNN,
    "VGGish": model_VGGish
}

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
val_directory =  "Project/val"

# Create datasets
val_dataset = MelSpecDataset(val_directory, class_mapping, transform)

# Create dataloaders
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)



# Assuming 'model' is your PyTorch model and 'val_loader' is your validation DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

for key, model in models.items():

    print("\n\nCLASSFICATION REPORT FOR MODEL: ", key, "\n\n")

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate precision, recall, and F1-score for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)

    # Print or store precision, recall, and F1-score for each class
    for i in range(len(precision)):
        print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}, F1-score={f1[i]}")

    print(f"Accuracy: {accuracy}")