import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from model_architectures import SimplifiedResNet, CNNModel, AttentionCNN, VGG10, VGGish  # Import other model architectures if needed
from preprocess import compute_mel_spectrogram, audio_preprocess, mono_to_color, apply_transform
import pandas as pd

# Define the directory paths
VAL_DATA_DIRECTORY_ABSOLUTE_PATH = "Project/val"
model_weights_paths = ['Project-Clone/Submission Files/simplifiedResNet_25epoch_weights.pth', 'Project-Clone/Model/CNNModel_10epochs_weights.pth', 'Project-Clone/Model/AttentionCNN.pth']  # Add paths for other models

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

# Define input shape and number of classes
input_shape = (128, 345, 3)
num_classes = 13

# Load models
models = [SimplifiedResNet(input_shape, num_classes), CNNModel(input_shape, num_classes), AttentionCNN(num_classes)]  # Instantiate all models

# Load validation dataset
ensemble_predictions = []
true_labels = []
    
    


# Define function to evaluate ensemble predictions
def evaluate_ensemble(file_path):
    predictions = []

    # Load and preprocess audio
    mel_spec = np.load(file_path)['mel_spec']
    mel_spec = apply_transform(mel_spec).unsqueeze(0)

    # Make predictions with each model
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(model_weights_paths[i]))
        model.eval()
        with torch.no_grad():
            output = model(mel_spec)
            _, predicted = torch.max(output, 1)
        predictions.append(predicted.item() + 1)  # Append prediction for current model to the list

    # Take a vote from the predictions
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction

# Iterate through validation dataset and compute predictions
for class_folder in os.listdir(VAL_DATA_DIRECTORY_ABSOLUTE_PATH):
    class_path = os.path.join(VAL_DATA_DIRECTORY_ABSOLUTE_PATH, class_folder)
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        true_label = class_mapping[class_folder]
        true_labels.append(true_label)
        ensemble_prediction = evaluate_ensemble(file_path)
        ensemble_predictions.append(ensemble_prediction)


# Compute metrics
conf_matrix = confusion_matrix(true_labels, ensemble_predictions)
precision = precision_score(true_labels, ensemble_predictions, average=None)
recall = recall_score(true_labels, ensemble_predictions, average=None)
f1 = f1_score(true_labels, ensemble_predictions, average=None)
accuracy = accuracy_score(true_labels, ensemble_predictions)

# Print or store precision, recall, and F1-score for each class
for i in range(len(precision)):
    print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}, F1-score={f1[i]}")

# Print accuracy
print(f"Accuracy: {accuracy}")
