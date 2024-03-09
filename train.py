"""
Author: Luka Secilmis

Description: Implements Conditional Variance Regularization (CoRe) on the MNIST dataset,
 following Section 5.5 of Heinze-Deml and N. Meinshausen's work on Conditional Variance Penalties and Domain Shift Robustness (https://arxiv.org/abs/1710.11469).

References: MNIST classification example (https://github.com/google/flax)
"""
from typing import Sequence

from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import datasets


class CNN(nn.Module):
    """Two-layer CNN as outlined in Table C.1."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x


def get_config():
    """Hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.learning_rate = 0.01
    config.batch_size = 120
    config.num_epochs = 25
    config.lambda_l2 = 0.001  # L2 regularization strength
    config.lambda_core = 0.2  # Conditional variance regularization strength
    return config


@jax.jit
def apply_model(state, images, labels, config):
    """Computes gradients, loss and accuracy for a single batch."""
    def core_penalty(params):  # TODO
        return config.lambda_l2 * 0

    def l2_penalty(params):
        return config.lambda_l2 * (jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params))

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)  # forward pass
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        loss += l2_penalty(params)
        loss += core_penalty(params)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # auxiliary data = logits
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, config, rng):
    """Train for one epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // config.batch_size

    # Shuffle dataset at start of each epoch
    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[: steps_per_epoch * config.batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels, config)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def create_train_state(rng, config):
    """Initialize weights."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> train_state.TrainState:
    """Trains the model on the MNIST dataset and evaluates it."""
    train_ds, test_ds = datasets.get_datasets()
    rng = jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds, config, input_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds['image'], test_ds['label'], config)

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,'
            ' test_accuracy: %.2f'
            % (
                epoch,
                train_loss,
                train_accuracy * 100,
                test_loss,
                test_accuracy * 100,
            )
        )

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test_loss, epoch)
        summary_writer.scalar('test_accuracy', test_accuracy, epoch)

    summary_writer.flush()
    return state
