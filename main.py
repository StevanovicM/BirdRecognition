# system libraries
import os
import time
import shutil
import pathlib
import itertools
from PIL import Image

#data handling tools
import cv2
import pandas as pd
import numpy as np
import seaborn as sbn
from keras.src.saving.saving_api import load_model

sbn.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#deep learning libraries

import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.applications.efficientnet import EfficientNetB0
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers.schedules import ExponentialDecay

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Read data and store it in dataframe
train_dir = 'train/'
filePaths = []
labels = []

folders = os.listdir(train_dir)
for folder in folders:
    folderPath = os.path.join(train_dir, folder)
    fileList = os.listdir(folderPath)
    for file in fileList:
        filePath = os.path.join(folderPath, file)
        filePaths.append(filePath)
        labels.append(folder)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filePaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
train_df = pd.concat([Fseries, Lseries], axis=1)

valid_dir = 'valid/'
filePaths = []
labels = []

folders = os.listdir(valid_dir)
for folder in folders:
    folderPath = os.path.join(valid_dir, folder)
    fileList = os.listdir(folderPath)
    for file in fileList:
        filePath = os.path.join(folderPath, file)
        filePaths.append(filePath)
        labels.append(folder)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filePaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
valid_df = pd.concat([Fseries, Lseries], axis=1)

test_dir = 'test/'
filePaths = []
labels = []

folders = os.listdir(test_dir)
for folder in folders:
    folderPath = os.path.join(test_dir, folder)
    fileList = os.listdir(folderPath)
    for file in fileList:
        filePath = os.path.join(folderPath, file)
        filePaths.append(filePath)
        labels.append(folder)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filePaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
test_df = pd.concat([Fseries, Lseries], axis=1)

# image size
batch_size = 32
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_datagen = ImageDataGenerator()
ts_datagen = ImageDataGenerator()
train_gen = tr_datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
valid_gen = ts_datagen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen = ts_datagen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                          class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

g_dict = train_gen.class_indices # Defines dictionary {'class': index}
classes = list(g_dict.keys()) # Defines list of dictionary's kays (classes), classes names : string
images, labels = next(train_gen)

plt.figure(figsize= (20, 20))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    image = images[i] / 255 # scales data to range (0 - 255)
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name = classes[index]
    plt.title(class_name, color='blue', fontsize=12)
    plt.axis('off')
plt.show()

# Create model structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys()))

# Load the pre-trained EfficientNetB0 model without the top layer (which is responsible for classification)
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

try:
    model = load_model('best_model.h5')
    print("Checkpoint loaded.")
except:
    print("No checkpoint found. Starting from scratch")
    # Create the model
    model = Sequential([
        base_model,
        BatchNormalization(axis= -1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(525, activation='softmax')
    ])
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

batch_size = 32 # Set batch size for training
epochs = 20 # number of epochs in training
history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen,
                    validation_steps=None, shuffle=False)

tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# ModelCheckpoint to save the model after every epoch
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# EarlyStopping to stop training when the validation loss has not improved after a certain number of epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

# Evaulate the model
train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)
print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])