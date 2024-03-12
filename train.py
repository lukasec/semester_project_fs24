"""
Author: Luka Secilmis

Description:
    Implements Conditional Variance Regularization (CoRe) on augmented MNIST dataset.

References:
    Reproduces experiment in section 5.5 of https://arxiv.org/abs/1710.11469 (C.Heinze-Deml and N. Meinshausen, 2017)
    Model architecture and hyperparameters adapted from https://github.com/christinaheinze/core/tree/master
    Adapted MNIST classification example from https://github.com/google/flax
"""
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import data_handler
SEED = 42

class CNN(nn.Module):
    """Two-layer CNN as outlined in Table C.1."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))(x)
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
    config.train_size = 10000  # Size of training set
    config.aug_size = 200  # Number of augmented images
    return config


@jax.jit
def apply_model(state, images, labels, ids, lambda_l2, lambda_core):
    """Computes gradients, loss and accuracy for a single batch."""
    def core_penalty(ids, logits, lambda_core):
        unique_ids = jnp.unique(ids)

        def mean_prediction_for_id(id):
            mask = ids == id  # indices of samples grouped by id
            return jnp.mean(logits[mask], axis=0)
        
        group_means = jax.vmap(mean_prediction_for_id)(unique_ids)
        
        def variance_for_id(id, group_mean):
            mask = ids == id
            group_logits = logits[mask]
            return jnp.mean(jnp.square(group_logits - group_mean)) if group_logits.shape[0] > 1 else 0
        
        group_variances = jax.vmap(variance_for_id, in_axes=(0, 0))(unique_ids, group_means)
        
        return lambda_core * jnp.mean(group_variances)

    def l2_penalty(params, lambda_l2):
        return lambda_l2 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params))

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)  # forward pass
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        loss += l2_penalty(params, lambda_l2)
        if ids is not None and lambda_core > 0: loss += core_penalty(ids, logits, lambda_core)
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
        batch_ids = train_ds['id'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels, batch_ids, config.lambda_l2, config.lambda_core)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def create_train_state(rng, learning_rate):
    """Initialize weights."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> train_state.TrainState:
    """Trains the model on the augmented MNIST dataset and evaluates it."""
    train_ds, test1_ds, test2_ds = data_handler.load_datasets("mnist", config.train_size, config.aug_size)
    rng = jax.random.key(SEED)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds, input_rng, config)
        _, test1_loss, test1_accuracy = apply_model(state, test1_ds['image'], test1_ds['label'], None, config.lambda_l2, config.lambda_core)
        _, test2_loss, test2_accuracy = apply_model(state, test2_ds['image'], test2_ds['label'], None, config.lambda_l2, config.lambda_core)

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test1_loss: %.4f,'
            ' test1_accuracy: %.2f', 'test2_loss: %.4f, test2_accuracy: %.2f'
            % (
                epoch,
                train_loss,
                train_accuracy * 100,
                test1_loss,
                test1_accuracy * 100,
                test2_loss,
                test2_accuracy * 100
            )
        )

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test1_loss, epoch)
        summary_writer.scalar('test_accuracy', test1_accuracy, epoch)
        summary_writer.scalar('test_loss', test2_loss, epoch)
        summary_writer.scalar('test_accuracy', test2_accuracy, epoch)

    summary_writer.flush()
    return state
