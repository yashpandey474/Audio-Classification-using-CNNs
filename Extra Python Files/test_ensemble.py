import os
import pandas as pd
import torch
import numpy as np
from model_architectures import SimplifiedResNet, CNNModel, AttentionCNN  # Import other model architectures if needed
from preprocess import compute_mel_spectrogram, audio_preprocess, mono_to_color, apply_transform

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "/Users/kmpandey/Desktop/3-2/Deep Learning/Project/val/Snare_drum"
OUTPUT_CSV_ABSOLUTE_PATH = "/Users/kmpandey/Desktop/3-2/Deep Learning/Project-Clone/outputensemble.csv"

# PARAMETERS PASSED TO MODELS
input_shape = (128, 345, 3)
num_classes = 13


# List of model weights paths
model_weights_paths = ['Project-Clone/Model/SimpleResNet_weights.pth', 'Project-Clone/Model/CNNModel_weights.pth', 'Project-Clone/Model/AttentionCNN.pth']  # Add paths for other models

def evaluate(file_path, models):
    predictions = []

    # # COMPUTE THE MEL SPECTOGRAM FOR THE AUDIO [FUNCTION PREPROCESSES AUDIO BEFORE CONVERSION]
    # mel_spec = compute_mel_spectrogram(file_path)
        
    # # CONVERT THE MEL SPECTOGRAM TO A 3-CHANNEL IMAGE
    # mel_spec = mono_to_color(mel_spec)

    # ONLY FOR OUR TESTING [REMOVE BEFORE SUBMISSION]
    mel_spec = np.load(file_path)['mel_spec']
    
    # APPLY TRANSFORMATIONS [CONVERTS TO TENSOR & OTHER TRANSFORMATIONS IF ANY]
    mel_spec = apply_transform(mel_spec).unsqueeze(0)
    
    for i, model in enumerate(models):
        # Load the trained weights
        model.load_state_dict(torch.load(model_weights_paths[i]))

        # Set the model to evaluation mode
        model.eval()

        # PREDICT THE CLASS
        with torch.no_grad():
            output = model(mel_spec)
            _, predicted = torch.max(output, 1) 

        predictions.append(predicted.item() + 1)  # Append prediction for current model to the list

    # Take a vote from the predictions
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction

def test():
    models = [SimplifiedResNet(input_shape, num_classes), CNNModel(input_shape, num_classes), AttentionCNN(num_classes)]  # Instantiate all models
    filenames = []
    final_predictions = []

    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        file_path = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
        final_prediction = evaluate(file_path, models)
        filenames.append(file_name)
        final_predictions.append(final_prediction)

    pd.DataFrame({"filename": filenames, "pred": final_predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)

# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
test()
