#!/usr/bin/env python

# """Description:
# The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
# This is just a simple template, you feel free to change it according to your own style.
# However, you must make sure:
# 1. Your own model is saved to the directory "model" and named as "model.h5"
# 2. The "test.py" must work properly with your model, this will be used by tutors for marking.
# 3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

# 2018 Created by Yiming Peng and Bing Xue
# """

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
#tf.set_random_seed(SEED)
tf.random.set_seed(SEED)

# no longer used
def create_dataset(directory):
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    subset= "training",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=20,
    image_size=(300, 300),
    shuffle=True,
    seed=309,
    validation_split=0.20,
    interpolation="bilinear",
    follow_links=False,)

    test_data =tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    subset= "validation",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=20,
    image_size=(300, 300),
    shuffle=True,
    seed=309,
    validation_split=0.20,
    interpolation="bilinear",
    follow_links=False,)



    return train_data, test_data

# generate datasets
def dataset_datagenerator(data_dir):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2,
        rotation_range=90,
        vertical_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=30,
        subset="training",
        class_mode='categorical',
        shuffle = True,
        classes = ['cherry', 'strawberry', 'tomato']
        )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=30,
        subset="validation",
        class_mode='categorical',
        shuffle = True,
        classes = ['cherry', 'strawberry', 'tomato']
        )

    return train_generator, validation_generator

def prefetching(train_data, test_data):
    # implementing buffered prefetching
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_data, test_data

def CNN_model2():
    num_classes = 3

    model = Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(strides=(2,2)),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),    

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),

    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', 
    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)),
    # tf.keras.layers.MaxPooling2D(strides=(2,2)),
    # tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )
    model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()
    return model
def CNN_model():
    num_classes = 3

    model = Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()
    return model

def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', input_dim=100),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation="softmax")
    ])
    # model.add(Dense(units=64, activation='relu', input_dim=100))
    # model.add(Dense(units=32, activation='relu'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, train_generator, validation_generator):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here
    epochs=95
    batch_size = 30
    # history = model.fit(
    # train_data,
    # validation_data=test_data,
    # epochs=epochs
    # )
    history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = epochs)

    # save_model(model)
    viz_results(history, epochs)

    return model

def viz_results(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("project/tf_proj/model/model1.h5")
    print("Model Saved Successfully.")

def pre_plot_imgs(train_data):
    class_names = train_data.class_names
    print(class_names)
    plt.figure(figsize=(10, 10))
    for images, labels in train_data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

if __name__ == '__main__':

    # without data generator - 1
    train_dir = "project/tf_proj/data/train_less_noise"
    # train_data, test_data = create_dataset(train_dir)
    # train_data, test_data = prefetching(train_data, test_data)
    # end - 1

    # with data generator - 2
    train_data, test_data = dataset_datagenerator(train_dir)
    # end - 2

    # pre_plot_imgs(train_data) # plot some images in the train set
    # model = construct_model()
    model = CNN_model2()
    model = train_model(model, train_data, test_data)


    save_model(model)
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))