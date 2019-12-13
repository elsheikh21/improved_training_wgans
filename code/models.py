import logging
import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Input, Conv2D, Dense, Activation,
                                     LeakyReLU, Flatten, BatchNormalization,
                                     Reshape, Dropout, Conv2DTranspose,
                                     UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (Adam, RMSprop)
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

from data_loader import load_dataset
from utils import initialize_logger, configure_tf, write_log


class GAN:
    def __init__(self, path, noise_dim=100, input_dim=(28, 28, 1), optimizer='adam_beta', batch_size=128, visualize=True):
        initialize_logger()
        configure_tf()
        self.path = path

        self.name = 'gan'
        self.input_dim = input_dim
        self.z_dim = noise_dim

        self.batch_size = batch_size
        self.train_x, self.train_y = load_dataset(batch_size=batch_size)
        logging.info('Dataset is loaded')

        self.optimizer = optimizer
        self.discriminator_lr = 8e-5
        self.generator_lr = 4e-5

        self.weight_initialization = RandomNormal(mean=0., stddev=0.02, seed=0)
        self.epochs = 5
        self.discriminator_losses, self.generator_losses = [], []

        self._build_discriminator_network()
        logging.info('Discriminator model is built')
        self._build_generator_network()
        logging.info('Generator model is built')
        self._build_adversarial_network()
        logging.info('GAN is built')

        if visualize:
            print(self.model.summary())
            print(self.generator.summary())
            print(self.discriminator.summary())

    def _build_discriminator_network(self):
        model_name = 'Discriminator'
        discriminator_input = Input(shape=self.input_dim, name=f'{model_name}_input')

        x = Conv2D(64, (5, 5), padding='same', input_shape=self.input_dim, name=f'{model_name}_Conv_1')(
            discriminator_input)
        x = LeakyReLU(alpha=0.2)(x)
        conv_layer1 = Dropout(0.2)(x)

        x_ = Conv2D(64, (5, 5), padding='same', name=f'{model_name}_Conv_2')(conv_layer1)
        x_ = LeakyReLU(alpha=0.2)(x_)
        conv_layer2 = Dropout(0.2)(x_)

        _x = Conv2D(128, (5, 5), padding='same', name=f'{model_name}Conv_3')(conv_layer2)
        _x = LeakyReLU(alpha=0.2)(_x)
        conv_layer3 = Dropout(0.2)(_x)

        _x_ = Conv2D(128, (5, 5), padding='same', name=f'{model_name}Conv_4')(conv_layer3)
        _x_ = LeakyReLU(alpha=0.2)(_x_)
        conv_layer4 = Dropout(0.2)(_x_)

        flat_output = Flatten()(conv_layer4)
        discriminator_output = Dense(1, activation='sigmoid',
                                     kernel_initializer=self.weight_initialization)(flat_output)
        self.discriminator = Model(inputs=discriminator_input, outputs=discriminator_output, name=model_name)

    def _build_generator_network(self):
        model_name = 'Generator'
        generator_input = Input(shape=(self.z_dim,), name='Generator_Input')
        x = Dense(7 * 7 * 256)(generator_input)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((7, 7, 256))(x)

        x = UpSampling2D()(x)
        x = Conv2D(filters=128, kernel_size=5, padding='same', name=f'generator_conv_1',
                   kernel_initializer=self.weight_initialization)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(64, kernel_size=5, padding='same', name=f'generator_conv_2',
                            kernel_initializer=self.weight_initialization)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(64, (5, 5), padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(1, (5, 5), padding='same', activation='tanh')(x)
        generator_output = Activation('tanh')(x)

        self.generator = Model(inputs=generator_input, outputs=generator_output, name=model_name)

    def _build_adversarial_network(self):
        self.discriminator.compile(loss='binary_crossentropy', metrics=['acc'],
                                   optimizer=self.get_optimizer(self.discriminator_lr))
        self.set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output, name=self.name)

        self.model.compile(loss='binary_crossentropy', metrics=['acc'],
                           optimizer=self.get_optimizer(self.generator_lr))
        self.set_trainable(self.discriminator, True)

    @staticmethod
    def set_trainable(model, is_trainable):
        model.trainable = is_trainable
        for layer_ in model.layers:
            layer_.trainable = is_trainable

    def plot_visualize_model(self):
        plot_model(self.model, to_file=os.path.join(
            self.path, 'visualize', 'model.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.discriminator, to_file=os.path.join(
            self.path, 'visualize', 'discriminator.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.generator, to_file=os.path.join(
            self.path, 'visualize', 'generator.png'), show_shapes=True, show_layer_names=True)

        with open(os.path.join(self.path, 'visualize', 'discriminator_summary.txt'), 'w+') as f:
            with redirect_stdout(f):
                self.discriminator.summary()
        with open(os.path.join(self.path, 'visualize', 'generator_summary.txt'), 'w+') as f:
            with redirect_stdout(f):
                self.generator.summary()
        with open(os.path.join(self.path, 'visualize', '/model_summary.txt'), 'w+') as f:
            with redirect_stdout(f):
                self.model.summary()

    def get_optimizer(self, learning_rate):
        if self.optimizer == 'adam_beta':
            return Adam(lr=learning_rate, beta_1=0.5)
        elif self.optimizer == 'rmsprop':
            return RMSprop(lr=learning_rate)
        else:
            return Adam(lr=learning_rate, beta_1=0.5)

    def train_discriminator_model(self):
        valid, fake = np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))
        idx = np.random.randint(0, self.train_x.shape[0], self.batch_size)
        true_data = self.train_x[idx]
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
        generated_imgs = self.generator.predict(noise)
        disc_real = self.discriminator.train_on_batch(true_data, valid)
        disc_fake = self.discriminator.train_on_batch(generated_imgs, fake)
        discriminator_loss = (disc_real[0] + disc_fake[0]) / 2
        discriminator_acc = (disc_real[1] + disc_fake[1]) / 2
        return [discriminator_loss, discriminator_acc]

    def train_generator_model(self):
        """
        Training generator on batches of the data, passing noise as the input
        for the generator as we defined our model earlier, our labels are ones.
        Train on batch returns a list of 2 params [loss, acc]
        """
        logs = self.model.train_on_batch(x=np.random.normal(0, 1, (self.batch_size, self.z_dim)),
                                         y=np.ones((self.batch_size, 1)))
        return [round(logs[0], 4), round(logs[1], 4)]

    def train(self, verbose=False, plot=True):
        log_path = os.path.join(self.path, 'logs')
        callback = TensorBoard(log_path)
        callback.set_model(self.model)
        for epoch in tqdm(range(1, self.epochs + 1), desc="GAN Training"):
            discriminator, generator = self.train_discriminator_model(), self.train_generator_model()
            self.discriminator_losses.append(discriminator)
            self.generator_losses.append(generator)
            write_log(callback, ['discriminator_loss', 'discriminator_acc', 'generator_loss'],
                      [discriminator[0], discriminator[1], generator[0]],
                      epoch)

            if verbose:
                logging.info(
                    f"Epoch number: {epoch} > Discriminator [loss: {discriminator[0]:.4f}, Acc: {discriminator[3]:.4f}%]. > Generator loss: {generator[0]:.4f}.")
        if plot:
            self._plot_training_history()

    def _plot_training_history(self):
        # loss Plot
        fig = plt.figure()
        plt.plot([x[0] for x in self.discriminator_losses], label='Discriminator Total loss')
        plt.plot([x[0] for x in self.generator_losses], label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        fig.legend()
        plt.xlim(0, self.epochs)
        plt.ylim(0, 2)
        plt.show()
        fig.savefig(os.path.join(self.path, 'visualize', 'gan_epochs_loss.png'))

        # Acc Plot
        fig = plt.figure()
        plt.plot([x[1] for x in self.discriminator_losses], label='Discriminator Total Acc')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        fig.legend()

        plt.xlim(0, self.epochs)

        plt.show()
        fig.savefig(os.path.join(self.path, 'visualize', 'gan_epochs_acc.png'))

    def save(self):
        self.model.save(os.path.join(self.path, 'weights', f'{self.name}_model.h5'))
        self.model.save_weights(os.path.join(self.path, 'weights', f'{self.name}_model_weights.h5'))

    def load_weights(self):
        self.model.load_weights(self.path)


class WGAN(GAN):
    def __init__(self, path, noise_dim=100, input_dim=(28, 28, 1),
                 optimizer='adam_beta', batch_size=128, visualize=True, clip_constant=0.01):
        super().__init__(path, noise_dim=100, input_dim=(28, 28, 1),
                         optimizer='adam_beta', batch_size=128, visualize=False)
        self.name = 'WGAN'
        self.train_x, self.train_y = load_dataset(label=5)
        self.clip_constant = clip_constant
        # Based on the ratio for training critic_epochs and generator_epoch
        self.critic_epochs = 5
        # Same network as GAN however this one should be
        # going with the name 'Critic' instead of discriminator
        self._build_discriminator_network()
        logging.info('Critic model is built')
        self._build_generator_network()
        logging.info('Generator model is built')
        self._build_adversarial_network()
        logging.info('WGAN is built')

    @staticmethod
    def wasserstein_loss(y_true, y_hat):
        return -1.0 * K.mean(y_true * y_hat)

    def _build_adversarial_network(self):
        self.discriminator.compile(optimizer=self.get_optimizer(self.discriminator_lr), loss=self.wasserstein_loss)
        self.set_trainable(self.discriminator, False)
        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)
        self.model.compile(optimizer=self.get_optimizer(self.generator_lr), loss=self.wasserstein_loss)
        self.set_trainable(self.discriminator, True)

    def train_discriminator_model(self):
        valid, fake = np.ones((self.batch_size, 1)), -np.ones((self.batch_size, 1))
        idx = np.random.randint(0, self.train_x.shape[0], self.batch_size)
        true_data = self.train_x[idx]
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
        generated_imgs = self.generator.predict(noise)

        disc_real = self.discriminator.train_on_batch(true_data, valid)
        disc_fake = self.discriminator.train_on_batch(generated_imgs, fake)
        discriminator_loss = (disc_real + disc_fake) / 2

        for layer_ in self.discriminator.layers:
            weights = layer_.get_weights()
            layer_.set_weights([np.clip(w, -self.clip_constant, self.clip_constant) for w in weights])
        return discriminator_loss

    def train_generator_model(self):
        """
        Training generator on batches of the data, passing noise as the input
        for the generator as we defined our model earlier, our labels are ones.
        Train on batch returns a list of 2 params [loss, acc]
        """
        return self.model.train_on_batch(x=np.random.normal(0, 1, (self.batch_size, self.z_dim)),
                                         y=np.ones((self.batch_size, 1)))

    def train(self, verbose=True, plot=True):
        log_path = os.path.join(self.path, 'logs')
        callback = TensorBoard(log_path)
        callback.set_model(self.model)
        for epoch in tqdm(range(1, self.epochs + 1), desc="WGAN Training"):
            for _ in range(self.critic_epochs):
                discriminator = self.train_discriminator_model()
                self.discriminator_losses.append(discriminator)
                write_log(callback, ['discriminator_loss'], [discriminator], epoch)
            generator = self.train_generator_model()
            self.generator_losses.append(generator)
            write_log(callback, ['generator_loss'], [generator], epoch)
            if verbose:
                logging.info(
                    f"Epoch number: {epoch} > Discriminator loss: {discriminator:.4f}. > Generator loss: {generator:.4f}.")
        if plot:
            self._plot_training_history()

    def _plot_training_history(self):
        # loss Plot
        fig = plt.figure()
        plt.plot(self.discriminator_losses, label='Discriminator loss')
        plt.xlabel('Epochs')
        plt.ylabel('DiscriminatorLoss')
        fig.legend()
        plt.show()
        fig.savefig(os.path.join(self.path, 'visualize', 'wgan_critic_loss.png'))

        fig = plt.figure()
        plt.plot(self.generator_losses, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Generator Loss')
        fig.legend()
        plt.show()
        fig.savefig(os.path.join(self.path, 'visualize', 'wgan_generator_loss.png'))


class WGAN_GP:
    pass
