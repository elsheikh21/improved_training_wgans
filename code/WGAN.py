import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tqdm import tqdm

from GAN import GAN
from data_loader import load_dataset
from utils import write_log


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
        plt.plot(self.generator_losses, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        fig.legend()
        plt.xlim(0, self.epochs)
        plt.ylim(0, 2)
        plt.show()
        fig.savefig(os.path.join(self.path, 'visualize', 'wgan_epochs_loss.png'))
