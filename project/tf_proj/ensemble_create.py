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
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from tensorflow.keras.preprocessing import image

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from  PIL import Image

def test_image(model, lst_images, pred_classes):
    print("called ====================================")
    for image in lst_images:            
        pred = modelEns.predict(image)
        print(pred)
        pred_classes.append(np.argmax(pred))
        print("prediciton: {}".format(pred.argmax(axis=-1)))
    
    
    return pred_classes

def int_classes(classes):
    y_test =[]
    for item in classes:
        if item == "cherry":
            y_test.append(0)
        if item == "strawberry":
            y_test.append(1)
        if item == "tomato":
            y_test.append(2)
    # y_test = np.array(classes)

    # # Binarize the labels
    # lb = LabelBinarizer()
    # y_test = lb.fit_transform(y_test)
    return y_test

def scan_folder(parent, all_classes, all_images, pred_classes):
    # iterate over all the files in directory 'parent'
    i = 0
    for file_name in os.listdir(parent):
        if file_name.endswith(".jpg"):
            # if it's a txt file, print its name (or do whatever you want)
            img = image.load_img("{}/{}".format(parent,file_name), target_size=(128, 128))
            img = image.img_to_array(img)
            i+=1
            # Get images and classes
            img = np.expand_dims(img, axis = 0)
            all_images.append(img)
            name = file_name.split("/")[-1]
            category = name.split("_")[-2]
            all_classes.append(category)

        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, all_classes, all_images, pred_classes)
        if i >= 21:
            break
        # print(all_classes)
    
    return all_images, all_classes, pred_classes

def ensemble_load(path):

    models=[]
    for i in range(1,5):
        full_path = "{}/model{}.h5".format(path,i)
        print(full_path, "--------------------")
        modelTemp=load_model(full_path) # load model
        print(modelTemp.summary())
        modelTemp._name="model{}.h5".format(i) # change name to be unique
        models.append(modelTemp)
    return models

def ensemble_models(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=tf.keras.layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  
    modelEns.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
   

    return modelEns

def gen_ensemble(models):
    model_input = Input(shape=models[0].input_shape[1:]) # c*h*w

    # model_input = Input([3, 128, 3])
    modelEns = ensemble_models(models, model_input)
    modelEns.summary()
    return modelEns

def load_ensemble_model(path):
    modelEns=load_model(path)

    modelEns.summary()
    return modelEns

def test_model(modelEns, x, y_true):
    correct_preds = 0
    pred_classes = []
    int_x = len(x)
    for i in range(len(x)):
        pred_classes.append(modelEns.predict(x[i]))

    for i in range(len(y_true)):
        if pred_classes[i] == y_true[i]:
            correct_preds += 1
    
    print("accuracy = {}".format(correct_preds/ int_x))

def compare_preds(pred_classes, true_classes):
    print(pred_classes)

    y_test = int_classes(true_classes)
    print(y_test)
    correct_preds = 0
    for i in range(len(true_classes)):
        if pred_classes[i] == y_test[i]:
            correct_preds += 1
    
    print("accuracy = {}".format(correct_preds/ len(pred_classes)))
    

def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """

    model.save("project/tf_proj/model/model5.h5")
    print("Model Saved Successfully.")

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
    model = load_model('project/tf_proj/model/model5.h5')
    print(model.summary())
    return model.evaluate(X_test, y_test, batch_size, verbose = 1)

def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    # X_test = np.array(images, dtype = None) #/ 255.0
    y_test = np.array(labels)
    print(y_test)
    # Binarize the labels
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return y_test

if __name__ == '__main__':
    all_classes = []
    all_images = []
    pred_classes = []
    models = ensemble_load("project/tf_proj/model")
    modelEns = gen_ensemble(models)

    save_model(modelEns)

    modelEns = load_ensemble_model("project/tf_proj/model/model5.h5")
    all_images, all_classes, pred_classes = scan_folder("project/tf_proj/data/test/cherry", all_classes, all_images, pred_classes)
    all_images, all_classes, pred_classes = scan_folder("project/tf_proj/data/test/strawberry", all_classes, all_images, pred_classes)
    all_images, all_classes, pred_classes = scan_folder("project/tf_proj/data/test/tomato", all_classes, all_images, pred_classes)

    # print(pred_classes)
    pred_classes = test_image(modelEns, all_images, pred_classes)
    compare_preds(pred_classes, all_classes)
    # # test_model(modelEns, data, classes)
    # binary_classes = convert_img_to_array(all_images, all_classes)

    # evaluate(all_images, binary_classes)
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))