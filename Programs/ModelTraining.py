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

History = Final_model.fit(
    x=training_batch_generator,
    steps_per_epoch=len(training_batch_generator),
    validation_data=val_batch_generator,
    validation_steps=len(val_batch_generator),
    epochs=50,
    callbacks=[lr_scheduler, visualize_callback, early_stopping, checkpoint_callback, save_logs_callback],
    verbose = 2
)

# Save the final model
Final_model.save('/content/drive/MyDrive/FinalExternal/Final Model/final_model.keras')
