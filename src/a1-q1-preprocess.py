from utils import timer, load_config, load_npz
import log
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from typing import Dict, List, Tuple

logger = log.get_logger(__name__)

@timer
def load_data():
    """
    Loads the CIFAR-10 dataset and returns the train and test sets.
    """
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    logger.info(f"Loaded CIFAR-10 dataset. Train shape: {trainX.shape}, Test shape: {testX.shape}")
    return (trainX, trainY), (testX, testY)

# @timer
# def save_npz(data: Dict[str, List[Tuple[int, int]]], filepath: str) -> None:
#     """
#     Save numpy array to compressed npz file.

#     Args:
#         data (Dict[str, List[Tuple[int, int]]]): A dictionary of numpy arrays to be saved.
#         filepath (str): The filepath to save the npz file to.
#     """
#     np.savez_compressed(filepath, **data)
#     logger.info(f"Saved npz file to {filepath}.")


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

@timer
def preprocess(conf):
    """
    Load the CIFAR10 dataset and split into training and validation sets.
    """
    data = load_npz(f"{conf.paths.a1_raw_data}cifar10.npz")
    logger.info(f"Loaded CIFAR10 dataset.")
    # split training data into training and validation sets
    trainX, trainY, valX, valY = split_dataset(data['trainX'], data['trainY'])
    # convert image data to float32 and normalize
    trainX, testX, valX = convert_image_data(trainX, data['testX'], valX)
    # one-hot encode labels
    trainY, testY, valY = encode_labels(trainY, data['testY'], valY)
    np.savez_compressed(f"{conf.paths.a1_input_data}cifar10.npz",
                        trainX=trainX, trainY=trainY, valX=valX, 
                     valY=valY, testX=testX, testY=testY)
    logger.info(f"Saved preprocessed data to {conf.paths.a1_input_data}cifar10.npz.")


if __name__ == "__main__":
    conf = load_config()
    preprocess(conf)