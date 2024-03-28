# CSF425 Deep Learning Project

## Description

File for testing: testing_code_template_new.py
File for training: training_plus_augmentation.py
File for preprocessing code: preprocess.py
File for model architectures: model_architectures.py

## NOTE: The following augmentation files only need to be run to replicate our models by training them on augmented training data
File for augmenting audio (1st step of augmenting): Augmentation - Audio.py
File for augmenting spectograms (2nd step of augmenting, after augmenting audios): Augmentation - Spectrogram.py

## STEPS FOR TESTING THE MODELS
For running the testing_code_template.py file, kindly follow the steps below

1. Change directory to the project folder
```cd CSF425 Deep Learning Project Folder```

2. Install dependencies using requirements.txt
```pip install -r requirements.txt```

3. Set the TEST_DATA_DIRECTORY_ABSOLUTE_PATH & OUTPUT_CSV_ABSOLUTE_PATH variables to absolute paths of test directory and where you want the output.csv to be created

4. Both model architectures are instantiated in the file. Currently, the uncommented evaluate uses the first architecture and the evaluate passing the second architecture is commented. Uncomment the evaluate line passing model_architecture2 and comment the one with model_architecture1 if you want to test the second architecture.

## FOLLOWING STEPS ARE ONLY FOR TRAINING USING AUGMENTED DATA [NOT FOR TESTING]

## STEPS FOR AUGMENTING DATA & TRAINING THE MODELS
1. In the training_plus_augmentation.py file, set the root_directory to the directory where the train and val folders for your audio.

2. Run the training_plus_augmentation.py file: ```python3 training_plus_augmentation.py```

3. The code will create new augmented audios in your train folder inside the root directory along with a new spectrograms folder in the root directory where it will store the spectrograms as .npz files

4. Then, it augments the spectrograms in this folder and saves them there

5. Next, it starts training using these spectrograms.

6. Please note, the second model architecture is commented in instantiation and can be uncommented if you wish to train that model instead of the first architecture.

7. The training displays the batch number and the epoch number along with training accuracy and val accuracy and returns the model with best val accuracy.

8. After training, the model weights will be available as: ```trained_model_weights.pth```

## STEPS FOR TRAINING DIRECTLY ON AUDIO WITHOUT AUGMENTATION  [OPTIONAL IF YOU DON'T WISH TO AUGMENT DATA]
1. In the training_on_audio.py file, set train_directory to your audio train folder and set val_directory to your audio val folder.

2. Run the training_on_audio.py file. It will show you the batch number it is on and the epoch number along with training accuracy and val accuracy and returns the model with best val accuracy.


## Team Members:
Yash Pandey, 2021A7PS0661P
Vikram Komperla, 2021A4PS1427P
Adhvik Ramesh, 2021AA2465P
