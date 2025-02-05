#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2020 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.preprocessing import image


# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
#tf.set_random_seed(SEED)
tf.random.set_seed(SEED)
print(tf.version.VERSION)

def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/test",
                      help = "path to test_data_dir")
    args = vars(args.parse_args())
    return args


def load_images(test_data_dir, image_size = (128, 128)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    imagePaths = list(paths.list_images(test_data_dir))
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    return images_data, labels 

def scan_folder(parent, all_classes, all_images, pred_classes):
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith(".jpg"):
            # if it's a txt file, print its name (or do whatever you want)
            img = image.load_img("{}/{}".format(parent,file_name), target_size=(128, 128))
            img = image.img_to_array(img)

            # Get images and classes
            all_images.append(img)
            name = file_name.split("/")[-1]
            category = name.split("_")[-2]
            all_classes.append(category)

        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # recall the method to check sub directories
                scan_folder(current_path, all_classes, all_images, pred_classes)
    
    return all_images, all_classes, pred_classes


def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_test = np.array(images, dtype = None) #/ 255.0
    y_test = np.array(labels)
    print(y_test)
    # Binarize the labels
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return X_test, y_test


def preprocess_data(X):
    """
    Pre-process the test data.
    :param X: the original data
    :return: the preprocess data
    """
    # NOTE: # If you have conducted any pre-processing on the image,
    # please implement this function to apply onto test images.
    return X

def evaluate(X_test, y_test):
    """
    Evaluation on test images
    **Please do not change this function**
    :param X_test: test images
    :param y_test: test labels
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model('project/tf_proj/model/model_standardNN.h5')
    print(model.summary())
    return model.evaluate(X_test, y_test, batch_size, verbose = 1)

if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["test_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (128, 128)

    # Load images
    # images, labels = load_images(test_data_dir, image_size)
    labels = []
    images = []
    pred_classes = []

    images, labels, pred_classes = scan_folder("project/tf_proj/data/test", labels, images, pred_classes)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    X_test, y_test = convert_img_to_array(images, labels)

    # # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    X_test = preprocess_data(X_test)
    #print(X_test.shape)
    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    print( y_test)
    loss, accuracy = evaluate(X_test, y_test)
    print("loss={}, accuracy={}".format(loss, accuracy))

    # modelEns = load_ensemble_model("project/tf_proj/model/model5.h5")

    # pred_classes = test_image(modelEns, y_test)
    # compare_preds(pred_classes, y_test)


