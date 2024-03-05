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

from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount('/content/drive')

# Path to the zip file in Google Drive
zip_file_path = '/content/drive/MyDrive/FinalExternal/mirflickr25k_preprocessed.zip'

# Directory to extract the zip file
extract_dir = '/content/Dataset/'

# Create the directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Zip file extracted successfully!")
