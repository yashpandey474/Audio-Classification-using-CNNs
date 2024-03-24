# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
from preprocess import compute_mel_spectrogram, audio_preprocess, mono_to_color, apply_transform
import torch
from model_architectures import SimplifiedResNet, AttentionCNN, CNNModel
import numpy as np


TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "Project/val/Snare_drum"
OUTPUT_CSV_ABSOLUTE_PATH = "Project-Clone/output.csv"
# The above two variables will be changed during testing. The current values are an example of what their contents would look like.
 # Instantiate the model
input_shape = (128, 345, 3)
num_classes = 13   
model_architecture1 = AttentionCNN(num_classes)

# Define the file path for the saved model weights
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path relative to the current directory
model_weights_path = os.path.join(current_dir, 'AttentionCNN.pth')

# Load the trained weights
model_architecture1.load_state_dict(torch.load(model_weights_path))

# Set the model to evaluation mode
model_architecture1.eval()

def evaluate(file_path, model):
    print("EVALUATING: ", file_path)
    # Write your code to predict class for a single audio file instance here
    
    # COMPUTE THE MEL SPECTOGRAM FOR THE AUDIO [FUNCTION PREPROCESSES AUDIO BEFORE CONVERSION]
    # mel_spec = compute_mel_spectrogram(file_path)
    
    # # CONVERT THE MEL SPECTOGRAM TO A 3-CHANNEL IMAGE
    # mel_spec = mono_to_color(mel_spec)


    # ONLY FOR OUR TESTING [REMOVE BEFORE SUBMISSION]
    mel_spec = np.load(file_path)['mel_spec']
    
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
    for file_path in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # TO USE ARCHITECTURE - 1
        prediction = evaluate(os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_path), model_architecture1)

        # TO USE ARCHITECTURE - 2
        # prediction = evaluate(os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_path), model_architecture2)
        filenames.append(file_path)
        predictions.append(prediction)

    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
test()
# test_batch()