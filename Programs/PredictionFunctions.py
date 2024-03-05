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

# Function to predict and visualize colorized images
def predict(image_path):
    predict_image_rgb = np.array((imread(image_path)))
    Encoder_image, Extractor_image, AB_channel = process(predict_image_rgb)

    fig = plt.figure(figsize=(20, 10))

    BW_image = rgb2gray(predict_image_rgb)
    Encoder_image = Encoder_image.reshape((1, 224, 224, 1))
    Extractor_image = Extractor_image.reshape((1, 299, 299, 3))
    predicted = Final_model.predict([[Encoder_image], [Extractor_image]])
    Encoder_image = Encoder_image * 50
    Encoder_image = Encoder_image + 50
    predicted = predicted * 128
    Final_predicted_image = np.concatenate((Encoder_image, predicted), axis=-1)
    Final_predicted_image = lab2rgb(Final_predicted_image[0])
    Final_predicted_image = skimage.transform.resize(Final_predicted_image, (predict_image_rgb.shape[0], predict_image_rgb.shape[1], 3), anti_aliasing=True)

    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('Input image')
    ax.imshow(BW_image, cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('Predicted image')
    ax.imshow(Final_predicted_image)
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('Ground Truth')
    ax.axis('off')
    ax.imshow(predict_image_rgb)

def predict_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            predict(image_path)
