import logging
import os
import random
import warnings

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.backend import set_session


def initialize_logger():
    """
    Customize the logger, and fixes seed
    """
    np.random.seed(0)
    random.seed(0)
    tf.set_random_seed(0)
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


def configure_tf():
    """
    Configuring TensorFlow GPU options, and TF logging messages
    """
    # Suppressing Warning messages and TF logging level
    warnings.filterwarnings('ignore', category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # Allowing TF to automatically choose an existing and
    # supported device to run the operations in case the specified one doesn't exist
    config.allow_soft_placement = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = False
    # Allocating a fraction from the GPU's memory
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)


def init_working_space(method):
    path = os.path.join(os.getcwd(), 'resources', f'{method}_running_dir')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'visualize'))
        os.makedirs(os.path.join(path, 'images'))
        os.makedirs(os.path.join(path, 'weights'))
    return path


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def load_config():
    config_file_path = os.path.join(os.getcwd(), "config.yaml")
    config_file = open(config_file_path)
    return yaml.load(config_file)
