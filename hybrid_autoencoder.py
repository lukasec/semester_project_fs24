import os
from typing import Sequence
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from flax.metrics import tensorboard
from absl import logging
import ml_collections
import data_handler
logging.set_verbosity(logging.INFO)
SEED = 7


class CNNEncoder(nn.Module):
    """Convolutional Encoder."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        x = nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.001)
        z = x.reshape((x.shape[0], -1))  # Flatten output
        return z


class CNNDecoder(nn.Module):
    """Convolutional Decoder."""
    output_shape: Sequence[int]
    @nn.compact
    def __call__(self, z):
        z = z.reshape((z.shape[0], 4, 4, 128))  # Reshape input for ConvTranspose
        z = nn.ConvTranspose(features=64, kernel_size=(5, 5), strides=(2, 2))(z)
        z = nn.leaky_relu(z, negative_slope=0.001)
        z = nn.ConvTranspose(features=32, kernel_size=(5, 5), strides=(2, 2))(z)
        z = nn.leaky_relu(z, negative_slope=0.001)
        z = nn.ConvTranspose(features=16, kernel_size=(5, 5), strides=(2, 2))(z)
        z = nn.leaky_relu(z, negative_slope=0.001)
        z = nn.ConvTranspose(features=3, kernel_size=(5, 5), strides=(2, 2))(z) 
        x_hat = nn.sigmoid(z)
        return x_hat
    

class HybridAutoEncoder(nn.Module):
    input_shape: Sequence[int]

    def setup(self):
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder(output_shape=self.input_shape)
        self.classifier = nn.Dense(features=1)

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return x_hat, logits
    

def apply_model(state, images, labels, lambda_aux):
    def loss_fn(params):
        x_hat, logits_aux = state.apply_fn({'params': params}, images)
        autoencoder_loss = jnp.mean((images - x_hat) ** 2)

        auxiliary_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_aux, labels))
        accuracy = jnp.mean((nn.sigmoid(logits_aux) > 0.5) == labels)

        loss = autoencoder_loss + lambda_aux * auxiliary_loss 
        return loss, (autoencoder_loss, auxiliary_loss, accuracy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (autoencoder_loss, auxiliary_loss, accuracy)), grads = grad_fn(state.params)
    return grads, loss, autoencoder_loss, auxiliary_loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, config, rng, lambda_aux):
    """Train for one epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // config.batch_size
    
    # Shuffle dataset at start of each epoch
    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[: steps_per_epoch * config.batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    epoch_loss = []
    epoch_ae_loss = []
    epoch_aux_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]

        grads, loss, autoencoder_loss, auxiliary_loss, accuracy = apply_model(state, batch_images, batch_labels, lambda_aux)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_ae_loss.append(autoencoder_loss)
        epoch_aux_loss.append(auxiliary_loss)
        epoch_accuracy.append(accuracy)

    train_loss = np.mean(epoch_loss)
    train_ae_loss = np.mean(epoch_ae_loss)
    train_aux_loss = np.mean(epoch_aux_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_ae_loss, train_aux_loss, train_accuracy


def create_train_state(rng, config):
    """Initialize weights and optimizer."""
    model = HybridAutoEncoder(input_shape=config.input_shape)
    params = model.init(rng, jnp.ones([1, 64, 64, 3]))['params']
    tx = optax.adam(learning_rate=config.learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, train_ds: dict) -> train_state.TrainState:
    rng = jax.random.key(SEED)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        # Train for one epoch
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_ae_loss, train_aux_loss, train_accuracy = train_epoch(state, train_ds, config, rng, config.lambda_aux)
        
        # Log results
        logging.info(
            f'epoch: {epoch}, train_loss: {train_loss:.4f}, '
            f'accuracy: {train_accuracy * 100:.2f}, '
            f'ae_loss: {train_ae_loss:.4f}, '
            f'aux_loss: {train_aux_loss:.4f}, '
            )

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_ae_loss', train_ae_loss, epoch)
        summary_writer.scalar('train_aux_loss', train_aux_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)

    summary_writer.flush()
    return state
