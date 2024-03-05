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

# Define model architectures
def IR_V2_model():
    IR_v2_base_model = InceptionResNetV2(weights='imagenet', include_top=True, input_shape=(299, 299, 3))
    IR_v2_output = RepeatVector(28*28)(IR_v2_base_model.output)
    IR_v2_output = Reshape((28, 28, 1000))(IR_v2_output)
    return Model(inputs=IR_v2_base_model.input, outputs=IR_v2_output)

def Encoder_model():
    model = Sequential(name="Encoder")
    model.add(InputLayer(input_shape=(224, 224, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=2))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    return model

def Decoder_model(fusion):
    model = Conv2D(256, (1, 1), activation="relu", padding="same", input_shape=(28, 28, 1256))(fusion)
    model = Conv2D(128, (3, 3), activation="relu", padding="same")(model)
    model = UpSampling2D((2, 2))(model)
    model = Conv2D(64, (3, 3), activation="relu", padding="same")(model)
    model = Conv2D(64, (3, 3), activation="relu", padding="same")(model)
    model = UpSampling2D((2, 2))(model)
    model = Conv2D(32, (3, 3), activation="relu", padding="same")(model)
    model = Conv2D(2, (3, 3), activation="tanh", padding="same")(model)
    model = UpSampling2D((2, 2))(model)
    return model

Extractor_path = IR_V2_model()
Extractor_path.trainable = False
Encoder_path = Encoder_model()
Fusion = tf.keras.layers.Concatenate(axis=-1)([Encoder_path.output, Extractor_path.output])
Decoder_output = Decoder_model(Fusion)
Final_model = Model([Encoder_path.input, Extractor_path.input], Decoder_output)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
Final_model.compile(optimizer=opt, loss='mse', metrics=['mae'])
