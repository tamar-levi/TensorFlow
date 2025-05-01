import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass

SEED_VALUE = 42

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)
print(X_test.shape)

# Plot CIFAR-10 images
plt.figure(figsize=(18, 8))

num_rows = 4
num_cols = 8

# plot each of the images in the batch and the associated ground truth labels.
for i in range(num_rows * num_cols):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(X_train[i, :, :])
    plt.axis("off")

# Normalize images to the range [0, 1].
X_train = X_train.astype("float32") / 255
X_test  = X_test.astype("float32")  / 255

# Convert labels to one-hot encoding.
print('Original (integer) label for the first training sample: ', y_train[0])

# Convert labels to one-hot encoding.
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

print('After conversion to categorical one-hot encoded labels: ', y_train[0])


@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 10
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 32
    NUM_CHANNELS: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS: int = 31
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 0.001


def cnn_model(input_shape=(32, 32, 3)):
    model = Sequential()

    # ------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    # ------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ------------------------------------
    # Flatten the convolutional features.
    # ------------------------------------
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# Create the model.
model = cnn_model()
model.summary()


# ---------------------------------------------------------
# Loading images from 'images' directory using ImageDataGenerator
# ---------------------------------------------------------

# Define the directory where your images are stored.
image_dir = 'images'

# Prepare the ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    os.path.join(image_dir, 'train'),  # Assuming your images are in 'images/train' directory
    target_size=(32, 32),  # Resizing images to (32, 32)
    batch_size=TrainingConfig.BATCH_SIZE,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    os.path.join(image_dir, 'validation'),  # Assuming your validation images are in 'images/validation'
    target_size=(32, 32),  # Resizing images to (32, 32)
    batch_size=TrainingConfig.BATCH_SIZE,
    class_mode='categorical')


# ---------------------------------------------------------
# Model Compilation and Training
# ---------------------------------------------------------

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // TrainingConfig.BATCH_SIZE,
    epochs=TrainingConfig.EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // TrainingConfig.BATCH_SIZE
)
