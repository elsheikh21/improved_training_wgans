from tensorflow.keras.datasets import mnist


def load_dataset(buffer_size=60000, batch_size=128, label=None):
    """
    Load MNIST Dataset in shape of (28, 28, 1), and return train_x, train_y
    """
    # Load the MNIST dataset
    (train_x, train_y), _ = mnist.load_data()
    if label is not None:
        desired_data = [y == label for y in train_y]
        train_x, train_y = train_x[desired_data], train_y[desired_data]

    # Normalize the images to [-1, +1] as per the paper,
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    train_x = (train_x - 127.5) / 127.5
    return train_x, train_y
