import logging
import os
import random
import warnings
import zipfile

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.backend import set_session
from tensorflow.keras.utils import get_file
from tqdm import tqdm


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


def download_unzip_dataset(url, save_to, clean_after=True):
    """
    Downloads the dataset and removes the zip file after unzipping it
    """
    file_path = get_file(fname=save_to, origin=url)
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        for file in tqdm(iterable=zip_file.namelist(),
                         total=len(zip_file.namelist()),
                         desc='Unzipping...'):
            zip_file.extract(member=file)
    if clean_after:
        os.remove(file_path)


if __name__ == '__main__':
    download_from = 'http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip'
    download_to = os.path.join(os.getcwd(), download_from.split('/')[-1])
    download_unzip_dataset(download_from, download_to, clean_after=False)
