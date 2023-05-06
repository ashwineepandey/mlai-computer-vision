import numpy as np
import utils as ut
import os
import log
import plotly.express as px
import pandas as pd
import itertools
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import gradient_descent_v2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
# from Typing import List, Tuple, Dict, Union

logger = log.get_logger(__name__)

@ut.timer
def load_fashion_mnist_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
    X_test = np.repeat(X_test[..., np.newaxis], 3, -1)
    
    X_train = np.asarray([tf.image.resize(img, (32, 32)).numpy() for img in X_train])
    X_test = np.asarray([tf.image.resize(img, (32, 32)).numpy() for img in X_test])
    
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test


@ut.timer
def load_pretrained_model(pretrained_model, weights, include_top=False):
    # Load pre-trained VGG16 model without the top layer (which includes the classification layers)
    base_model = pretrained_model(weights=weights, include_top=False, input_shape=(32, 32, 3))
    return base_model

@ut.timer
def add_new_layers(base_model, dense_units, activation):
    # Add new layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(units=dense_units, activation=activation)(x)
    predictions = Dense(10, activation='softmax')(x) # For CIFAR10 data
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

@ut.timer
def freeze_layers(base_model):
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    _check_trainable_layers(base_model)
    return base_model

@ut.timer
def unfreeze_layers(base_model):
    # Unfreeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = True
    _check_trainable_layers(base_model)
    return base_model


def _check_trainable_layers(model):
    # get the configuration of each layer
    layer_configs = model.get_config()['layers']

    # print whether each layer is trainable or not
    for i, layer_config in enumerate(layer_configs):
        layer = model.layers[i]
        logger.info(f"Layer {i}: {layer.name}, trainable={layer.trainable}")

@ut.timer
def compile_model(model, loss, metrics, learning_rate=0.0001, momentum=0.9):
    opt = gradient_descent_v2.SGD(learning_rate=learning_rate, momentum=momentum)
    # We need to recompile the model for these modifications to take effect
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

@ut.timer
def train_model(model: Model, 
                trainX: np.ndarray, 
                trainY: np.ndarray, 
                valX: np.ndarray, 
                valY: np.ndarray, 
                batch_size: int, 
                epochs: int, 
                model_path: str, 
                model_name: str, 
                datetime: str) -> History:
    """
    Trains the CNN model on the training set and validates on the validation set.
    """
    # define the callback to save the weights
    checkpoint = ModelCheckpoint(f'{model_path}{model_name}_classifier_{datetime}.h5', 
                                 monitor='val_accuracy', 
                                 save_best_only=True, 
                                 save_weights_only=True, 
                                 verbose=1)

    history = model.fit(trainX, trainY,
                        validation_data=(valX, valY),
                        callbacks=[checkpoint], 
                        batch_size=batch_size, 
                        epochs=epochs,
                        verbose=1)    
    return model, history

@ut.timer
def main():
    # load config
    conf = ut.load_config()
    # load data
    trainX, trainY, testX, testY = load_fashion_mnist_data()
    # split training data into training and validation sets
    trainX, trainY, valX, valY = ut.split_dataset(trainX, trainY)
    # convert image data to float32 and normalize
    trainX, testX, valX = ut.convert_image_data(trainX, testX, valX)
    # get datetime
    datetime = ut.get_current_dt()
    # load pre-trained model
    base_model = load_pretrained_model(pretrained_model=VGG16, weights='imagenet', include_top=False)
    # freeze layers
    freeze_layers(base_model)
    # add new layers
    model = add_new_layers(base_model, dense_units=512, activation='relu')
    # compile model
    compile_model(model, loss='categorical_crossentropy', metrics=['accuracy'],
                  learning_rate=conf.a3.classifier_params.learning_rate[0], 
                  momentum=conf.a3.classifier_params.momentum[0])
    # train model
    _, history = train_model(model, trainX, trainY, valX, valY, 
                        batch_size=conf.a3.classifier_params.batch_size, 
                        epochs=conf.a3.classifier_params.epochs,
                        model_path=conf.a3.paths.model, 
                        model_name="vgg16", 
                        datetime=datetime)
    # unfreeze layers
    unfreeze_layers(base_model)
    # compile model
    compile_model(model, loss='categorical_crossentropy', metrics=['accuracy'])
    # train model
    model, history = train_model(model, trainX, trainY, valX, valY, 
                        batch_size=conf.a3.classifier_params.batch_size, 
                        epochs=conf.a3.classifier_params.epochs,
                        model_path=conf.a3.paths.model, 
                        model_name="vgg16", 
                        datetime=datetime)
    # Save the trained model
    model.save(f"{conf.a3.paths.model}fashion_mnist_classifier_{datetime}.h5")
    # save trained model and training history
    ut.save_history(conf.a3.paths.train_history, history, "fashionMNIST", "", datetime)
    fig = ut.plot_performance(history, dataset="validation")
    ut.save_plot(conf.a3.paths.train_plots, fig, "validation", datetime)


if __name__ == "__main__":
    main()