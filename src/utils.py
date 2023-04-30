import sys
import os
import yaml
import log
import functools
import time
from pyhocon import ConfigFactory
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
from numpy import savez_compressed, load
import keras.backend as K
from keras.utils import np_utils
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

logger = log.get_logger(__name__)

def timer(func):
    """ Print the runtime of the decorated function """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logger.info(f"Starting {func.__name__!r}.")
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs.")
        return value
    return wrapper_timer


def debug(func):
    """ Print the function signature and return value """
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        logger.debug(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


@timer
def load_config():
    """
    Load the configuration file from a hocon file object and returns it (https://github.com/chimpler/pyhocon).
    """
    return ConfigFactory.parse_file("../conf/main.conf")

@timer
def save_npz(data: Dict[str, List[Tuple[int, int]]], filepath: str) -> None:
    """
    Save numpy array to compressed npz file.

    Args:
        data (Dict[str, List[Tuple[int, int]]]): A dictionary of numpy arrays to be saved.
        filepath (str): The filepath to save the npz file to.
    """
    savez_compressed(filepath, **data)
    logger.info(f"Saved npz file to {filepath}.")

@timer
def load_npz(filepath: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Load numpy array from compressed npz file.

    Args:
        filepath (str): The filepath to load the npz file from.

    Returns:
        Dict[str, List[Tuple[int, int]]]: A dictionary of numpy arrays.
    """
    data = load(filepath)
    logger.info(f"Loaded npz file from {filepath}.")
    return data

@timer
def _shuffle_data(trainX, trainY):
    # shuffle dataset
    shuffle_idx = np.random.permutation(len(trainX))
    trainX, trainY = trainX[shuffle_idx], trainY[shuffle_idx]
    return trainX, trainY

@timer
def split_dataset(trainX, trainY):
    """
    Split the training dataset into training and validation datasets.
    """
    trainX, trainY = _shuffle_data(trainX, trainY)
    # split into 40000 and 10000
    trainX, valX = trainX[:40000], trainX[40000:]
    trainY, valY = trainY[:40000], trainY[40000:]
    return trainX, trainY, valX, valY

@timer
def convert_image_data(trainX, testX, valX, float_type='float32', norm_val=255.0):
    """
    Converts pixel values from integers to floats and normalizes to range 0-1.
    """
    # convert from integers to floats and normalize to range 0-1
    trainX = trainX.astype(float_type) / norm_val
    testX = testX.astype(float_type) / norm_val
    valX = valX.astype(float_type) / norm_val
    return trainX, testX, valX

@timer
def encode_labels(trainY, testY, valY):
    """
    One-hot encodes the labels.
    """
    trainY = np_utils.to_categorical(trainY) # e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    testY = np_utils.to_categorical(testY)
    valY = np_utils.to_categorical(valY)
    return trainY, testY, valY


def sizeof(obj):
    """
    Get size of object in memory.
    """
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    logger.info(f"Size in memory =  {size} B = {size / 1000000} MB.")


def read_yaml(config_path: str) -> dict:
    """
    Reads yaml file and returns as a python dict.

    Args:
        config_path (str) : Filepath of yaml file location.

    Returns:
        dict: A dictionary of the yaml filepath parsed in.
    """
    with open(config_path, "r") as f:
        logger.info(f"Config file read in successfully.")
        return yaml.safe_load(f)


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0.0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


@timer
def save_plot(plot_filepath, fig, dataset, datetime):
    """
    Saves the plot to file.
    """
    # generate file name
    prefix = f"cifar10-classification-{dataset}-{datetime}.png"
    filepath = os.path.join(plot_filepath, prefix)
    # save figure to file
    fig.write_image(filepath)
    logger.info(f"Plot saved: {prefix}")


@timer
def save_model(model_path, model, model_name, combination_num, datetime):
    """
    Saves the trained model to disk.
    """
    # Create directory for saving models if it does not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Save model architecture and weights
    model.save(f"{model_path}{model_name}_{combination_num}_{datetime}.h5")
    logger.info(f"Model saved: {model_path}{model_name}_{combination_num}_{datetime}.h5")

@timer
def save_history(history_path, history, model_name, combination_num, datetime):
    """
    Saves the training history to disk.
    """
    # Create directory for saving history if it does not exist
    if not os.path.exists(history_path):
        os.makedirs(history_path)
    
    # Save training history to csv
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"{history_path}{model_name}_{combination_num}_{datetime}_history.csv", index=False)
    logger.info(f"Training history saved: {history_path}{model_name}_{combination_num}_{datetime}_history.csv")


def get_current_dt() -> str:
    """
    Returns the current date and time as a string in the format "DDMMYYYY_HHMMSS".

    Returns:
        A string representing the current date and time, in the format "DDMMYYYY_HHMMSS".
    """
    now = datetime.now()
    return now.strftime("%d%m%Y_%H%M%S")


@timer
def plot_performance(history, dataset):
    """
    Plots the loss and accuracy curves and saves the figure to file.
    """
    # create figure with two subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Cross Entropy Loss', 'Classification Accuracy'))
    # plot loss
    fig.add_trace(px.line(history.history, x=history.epoch, y=['loss', 'val_loss'], labels={'value': 'loss', 'variable': ''}).data[0], row=1, col=1)
    # plot accuracy
    fig.add_trace(px.line(history.history, x=history.epoch, y=['accuracy', 'val_accuracy'], labels={'value': 'accuracy', 'variable': ''}).data[0], row=2, col=1)
    # add final metrics as text
    final_loss = round(history.history['loss'][-1], 4)
    final_val_loss = round(history.history['val_loss'][-1], 4)
    final_acc = round(history.history['accuracy'][-1], 4)
    final_val_acc = round(history.history['val_accuracy'][-1], 4)
    fig.add_annotation(text=f"Final Loss: {final_loss}\nFinal Val Loss: {final_val_loss}", xref="paper", yref="paper", x=0.02, y=0.95, showarrow=False)
    fig.add_annotation(text=f"Final Accuracy: {final_acc}\nFinal Val Accuracy: {final_val_acc}", xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False)
    # update layout
    fig.update_layout(title=f'{dataset.capitalize()} CIFAR-10 Classification', width=800, height=600)
    return fig