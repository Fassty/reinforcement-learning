import glob
import os
import time
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, \
    Conv2D, Conv2DTranspose, Reshape
import numpy as np
import gym
import cart_pole_pixels_environment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CVAE(tf.keras.Model):
    def __init__(self, observation_shape, latent_dim, seed):
        super().__init__()
        self.seed = seed
        self.latent_dim = latent_dim

        encoder_in = tf.keras.Input(shape=observation_shape)  # 80 x 80 x 3
        x = Conv2D(32, kernel_size=4, strides=3, padding='valid', activation='relu')(encoder_in)  # 39 x 39 x 32
        x = Conv2D(64, kernel_size=4, strides=2, padding='valid', activation='relu')(x)  # 18 x 18 x 64
        x = Conv2D(128, kernel_size=4, strides=2, padding='valid', activation='relu')(x)  # 8 x 8 x 128
        x = Conv2D(256, kernel_size=4, strides=2, padding='valid', activation='relu')(x)  # 3 x 3 x 256
        x = Flatten()(x)
        z_mu = Dense(self.latent_dim)(x)
        z_logvar = Dense(self.latent_dim)(x)

        self.encoder = tf.keras.Model(inputs=encoder_in, outputs=[z_mu, z_logvar])

        decoder_in = tf.keras.Input(shape=[self.latent_dim])
        x = Dense(5 * 5 * 128, activation='relu')(decoder_in)
        x = Reshape(target_shape=(5, 5, 128))(x)
        x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        decoder_out = Conv2DTranspose(3, kernel_size=4, strides=1, padding='same')(x)

        self.decoder = tf.keras.Model(inputs=decoder_in, outputs=decoder_out)

        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.writer = tf.summary.create_file_writer('logdir')

    @tf.function
    def sample(self, batch_size, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed)
        return self.decode(eps, apply_sigmoid=True)

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def save_VAE(self):
        self.encoder.save_weights('encoder_weights')
        self.decoder.save_weights('decoder_weights')

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape, seed=self.seed)
        return eps * tf.exp(logvar * .5) + mu

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def get_latent_representation(self, sample):
        mu, logvar = self.encode(sample)
        return self.reparameterize(mu, logvar)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def data_iterator(batch_size):
    data_files = glob.glob('./data/obs_data_VAE_*')
    while True:
        data = np.load(random.sample(data_files, 1)[0])
        np.random.shuffle(data)
        np.random.shuffle(data)
        N = data.shape[0]
        start = np.random.randint(0, N - batch_size)
        yield data[start:start + batch_size]


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1))
    return images


env = gym.make('CartPolePixels-v0')

def load_vae():
    model = CVAE(env.observation_space.shape, 4, 42)

    model.encoder.load_weights('./encoder_weights')
    model.decoder.load_weights('./decoder_weights')

    return model


if __name__ == '__main__':
    _LATENT_DIM = 4
    _SEED = 42
    _EPOCHS = 100000
    _BATCH_SIZE = 128

    model = CVAE(env.observation_space.shape, _LATENT_DIM, _SEED)

    training_data = data_iterator(_BATCH_SIZE)
    for i in range(_EPOCHS):
        images = next(training_data)

        start_time = time.time()
        train_step(model, images)
        end_time = time.time()

        test_images = next(training_data)
        loss = tf.keras.metrics.Mean()
        loss(compute_loss(model, images))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(i, elbo, end_time - start_time))

        if i % 10 == 0:
            model.save_VAE()
