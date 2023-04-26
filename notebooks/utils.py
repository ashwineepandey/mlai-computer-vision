import sys
import yaml
import log
import functools
import time
from pyhocon import ConfigFactory
from typing import Dict, List, Tuple
from datetime import datetime
from numpy import savez_compressed, load
import keras.backend as K


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


def get_current_dt() -> str:
    """
    Returns the current date and time as a string in the format "DDMMYYYY_HHMMSS".

    Returns:
        A string representing the current date and time, in the format "DDMMYYYY_HHMMSS".
    """
    now = datetime.now()
    return now.strftime("%d%m%Y_%H%M%S")