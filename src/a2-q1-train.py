import numpy as np
import utils as ut
import os
import log
import plotly.express as px
import pandas as pd
import itertools
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import gradient_descent_v2
from keras.applications.vgg16 import VGG16
# from Typing import List, Tuple, Dict, Union

logger = log.get_logger(__name__)

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
                combination_num: int, 
                datetime: str) -> History:
    """
    Trains the CNN model on the training set and validates on the validation set.
    """
    # define the callback to save the weights
    checkpoint = ModelCheckpoint(f'{model_path}{model_name}_{combination_num}_{datetime}.h5', 
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
    return history

@ut.timer
def main():
    # load config
    conf = ut.load_config()
    # load data
    data = ut.load_npz(f"{conf.paths.a1_raw_data}cifar10.npz")
    # split training data into training and validation sets
    trainX, trainY, valX, valY = ut.split_dataset(data['trainX'], data['trainY'])
    # convert image data to float32 and normalize
    trainX, _, valX = ut.convert_image_data(trainX, data['testX'], valX)
    # one-hot encode labels
    trainY, _, valY = ut.encode_labels(trainY, data['testY'], valY)
    # get datetime
    datetime = ut.get_current_dt()
    # load pre-trained model
    base_model = load_pretrained_model(pretrained_model=VGG16, weights='imagenet', include_top=False)
    # freeze layers
    freeze_layers(base_model)
    # add new layers
    model = add_new_layers(base_model, dense_units=512, activation='relu')
    # compile model
    compile_model(model, loss='categorical_crossentropy', metrics=['accuracy'])
    # train model
    history = train_model(model, trainX, trainY, valX, valY, batch_size=32, epochs=10,
                            model_path=conf.paths.a2_model, 
                            model_name="vgg16", 
                            combination_num=0, 
                            datetime=datetime)
    # unfreeze layers
    unfreeze_layers(base_model)
    # compile model
    compile_model(model, loss='categorical_crossentropy', metrics=['accuracy'])
    # train model
    history = train_model(model, trainX, trainY, valX, valY, batch_size=32, epochs=10,
                            model_path=conf.paths.a2_model, model_name="vgg16", combination_num="base_model", datetime=datetime)
    # save trained model and training history
    ut.save_history(conf.paths.a2_train_history, history, "cifar10", "", datetime)
    fig = ut.plot_performance(history, dataset="validation")
    ut.save_plot(conf.paths.a2_train_plots, fig, "validation", datetime)


if __name__ == "__main__":
    main()