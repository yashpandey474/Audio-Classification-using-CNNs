# Audio Classification using Convolutional Neural Networks
A project to classify audio belonging to 13 different classes using a convolutional neural network (CNN). This project was built as part of our CSF425 Deep Learning course and our team obtained the highest accuracy of 89% on a held-out test set.

# Project Description
We have experimented with various types of CNN architectures such as ResNet-based CNNs, LSTM-based CNNs, Attention-based CNNs and VGGish for classifying audios belonging to 13 different classes.
We were provided with a dataset within which we noticed the following problems:
1. Class imbalance
2. Different formats of audio
3. Varying degrees of background noise
4. Different length of audio files

We have conducted the following major experiments:
1. Data augmentation to correct the class imbalance by setting a 'per-file-augmentation-factor' and then randomly selecting files for augmentations.
2. Trimmed & normalised the audio files to bring them to a common format.
3. Trained different architectures and evaluated their classification metrics with respect to each class along with overall performance in terms of precision, recall and accuracy.
4. Conducted hyperparameter optimisation experiments on our chosen, ResNet CNN
5. Compared results with respect to training on augmeneted data versus training on non-augmented data, using ResNet CNN

# Running the project
The submission file folder contains code that can be used for testing the trained models.

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
