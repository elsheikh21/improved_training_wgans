import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def load_celeb_a(image_size=64, batch_size=64):
    path = os.path.join(os.getcwd(), 'data')
    data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5)
    train_x = data_gen.flow_from_directory(path, target_size=(image_size, image_size), batch_size=batch_size,
                                           class_mode='input', subset="training", seed=0)
    return train_x
