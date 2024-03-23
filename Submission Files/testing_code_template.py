# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
from preprocess import compute_mel_spectrogram, audio_preprocess, mono_to_color, apply_transform
import torch

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "/home/pc/test_data"
OUTPUT_CSV_ABSOLUTE_PATH = "/home/pc/output.csv"
# The above two variables will be changed during testing. The current values are an example of what their contents would look like.
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your layers here
        self.fc = nn.Linear(10, 2)  # Example linear layer

    def forward(self, x):
        # Define the forward pass
        x = self.fc(x)
        return x

 # Instantiate the model
model = MyModel()

# Define the file path for the saved model weights
model_weights_path = 'model_weights.pth'

# Load the trained weights
model.load_state_dict(torch.load(model_weights_path))

# Set the model to evaluation mode
model.eval()

def evaluate(file_path):
    # Write your code to predict class for a single audio file instance here

    # COMPUTE THE MEL SPECTOGRAM FOR THE AUDIO [FUNCTION PREPROCESSES AUDIO BEFORE CONVERSION]
    mel_spec = compute_mel_spectrogram(file_path)
    
    # CONVERT THE MEL SPECTOGRAM TO A 3-CHANNEL IMAGE
    mel_spec = mono_to_color(mel_spec)

    # APPLY TRANSFORMATIONS [CONVERTS TO TENSOR & OTHER TRANSFORMATIONS IF ANY]
    mel_spec = apply_transform(mel_spec)

    # PREDICT THE CLASS
    with torch.no_grad():
        output = model(mel_spec)
        _, predicted = torch.max(output, 1) 

    return predicted + 1


# def evaluate_batch(file_path_batch, batch_size=32):
#     # Write your code to predict class for a batch of audio file instances here
#     return predicted_class_batch


def test():
    filenames = []
    predictions = []
    for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        prediction = evaluate(file_path)

        filenames.append(file_path)
        predictions.append(prediction)
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


def test_batch(batch_size=32):
    filenames = []
    predictions = []

    paths = os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    
    # Iterate over the batches
    # For each batch, execute evaluate_batch function & append the filenames for that batch in the filenames list and the corresponding predictions in the predictions list.
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
# test()
# test_batch()