import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import skimage.transform
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Dense, InputLayer, Concatenate, Reshape, RepeatVector
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import cv2

import random

# Set a random seed for reproducibility
random.seed(42)

# Load and preprocess image filenames
train_directory = '/content/Dataset/mirflickr25k_preprocessed/Train'
All_filenames = []
for file in sorted(os.listdir(train_directory), key=len):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        ab = np.array(imread((train_directory+'/'+filename)))
        if ab.shape[-1] == 3:
            All_filenames.append(filename)

# Shuffle the filenames with the same seed
random.shuffle(All_filenames)

# Define the ratio for the validation set (e.g., 20%)
val_ratio = 0.2

# Calculate the number of samples for the validation set
val_size = int(len(All_filenames) * val_ratio)


# Split the filenames into training and validation sets
train_filenames = All_filenames[val_size:]
val_filenames = All_filenames[:val_size]
print("Number of images in training set:", len(train_filenames))
print("Number of images in validation set:", len(val_filenames))
