import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import download_unzip_dataset


def load_mnist(label=None):
    """
    Load MNIST Dataset in shape of (28, 28, 1), and return train_x, train_y
    """
    # Load the MNIST dataset
    (train_x, train_y), _ = mnist.load_data()

    if label is not None:
        desired_data = [y == label for y in train_y]
        train_x, train_y = train_x[desired_data], train_y[desired_data]

    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images to [-1, +1] as per the paper,
    train_x = (train_x - 127.5) / 127.5
    return train_x, train_y


def load_lsun(image_size=64, batch_size=64):
    try:
        # TODO: Add the name of the file to load
        # TODO: Know how to export the Data from mdb format
        path = os.path.join(os.getcwd(), 'data', '')
        assert os.path.exists(path)
    except AssertionError:
        download_from = 'http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip'
        download_to = os.path.join(os.getcwd(), 'data', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to, clean_after=False)
        path = download_to

    data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5)
    train_x = data_gen.flow_from_directory(path, target_size=(image_size, image_size), batch_size=batch_size,
                                           class_mode='input', seed=0)
    return train_x, None
