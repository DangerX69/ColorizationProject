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

# Define the model architecture
Extractor_path = IR_V2_model()
Extractor_path.trainable = False
Encoder_path = Encoder_model()
Fusion = tf.keras.layers.Concatenate(axis=-1)([Encoder_path.output, Extractor_path.output])
Decoder_output = Decoder_model(Fusion)
Final_model = Model([Encoder_path.input, Extractor_path.input], Decoder_output)

Final_model = tf.keras.models.load_model('/content/final_model')

predict_folder('/content/Dataset/human_face/Test/')
