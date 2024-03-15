"""
Author: Luka Secilmis

Description:
    Implements Conditional Variance Regularization (CoRe) on augmented MNIST dataset.

References:
    Reproduces experiment in section 5.5 of https://arxiv.org/abs/1710.11469 (C.Heinze-Deml and N. Meinshausen, 2017)
    Model architecture and hyperparameters adapted from https://github.com/christinaheinze/core/tree/master
    Adapted MNIST classification example from https://github.com/google/flax
"""
import os
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
logging.set_verbosity(logging.INFO)
SEED = 7

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


def apply_model(state, images, labels, ids, lambda_l2, lambda_core):
    """Computes gradients, loss and accuracy for a single batch."""
    def core_penalty(params, logits, ids):
        _, index_inverse, counts = jnp.unique(ids, return_inverse=True, return_counts=True)  # ids = unique_ids[index_inverse]
        
        mask = counts[index_inverse] > 1
        
        penalty = 0.0
        
        if jnp.any(mask):
            # IDs with more than one occurence
            filtered_logits = logits[mask] 
            
            # Mean logits per ID
            sum_perID = jax.ops.segment_sum(filtered_logits, index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            count_perID = jax.ops.segment_sum(jnp.ones_like(filtered_logits[:, 0]), index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            mean_perID = sum_perID / count_perID[:, None]
            
            # Squared differences between logits and their mean
            square_diff = (filtered_logits - mean_perID[index_inverse[mask]])**2
            
            # Variance per ID
            sum_squares_perID = jax.ops.segment_sum(square_diff, index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            variance_perID = sum_squares_perID / count_perID[:, None]
            
            # Mean of the variances per ID
            penalty = jnp.mean(variance_perID[count_perID > 1])
        return penalty
        
    def l2_penalty(params):
        return sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params))

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)  # forward pass
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        l2_pen = l2_penalty(params)
        loss += lambda_l2 * l2_pen
        core_pen = core_penalty(params, logits, ids)
        loss += lambda_core * core_pen
        return loss, (logits, core_pen)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # auxiliary data
    (loss, (logits, core_pen)), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy, core_pen


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
    core_penalties= []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        
        # for CoRe penalty
        batch_ids = train_ds['id'][perm, ...]
        # _, index_inverse, batch_counts = jnp.unique(batch_ids, return_inverse=True, return_counts=True)  # ids = unique_ids[index_inverse]
        # batch_num_segments = jnp.max(index_inverse) + 1
        # batch_mask = batch_counts[index_inverse] > 1
        # masked_inverse = index_inverse[batch_mask]

        grads, loss, accuracy, core_pen = apply_model(state, batch_images, batch_labels, batch_ids, config.lambda_l2, config.lambda_core)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
        core_penalties.append(core_pen)

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    core_penalty= np.mean(core_penalties)
    return state, train_loss, train_accuracy, core_penalty


def create_train_state(rng, learning_rate, decay_rate, decay_steps):
    """Initialize weights."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    schedule = optax.exponential_decay(init_value=learning_rate,
                                       transition_steps=decay_steps,
                                       decay_rate=decay_rate)
    tx = optax.adam(schedule)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, train_ds: dict, test1_ds: dict, test2_ds: dict) -> train_state.TrainState:
    rng = jax.random.key(SEED)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    decay_steps = len(train_ds['image']) // config.batch_size
    state = create_train_state(init_rng, config.learning_rate, config.decay_rate, decay_steps)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy, core_penalty = train_epoch(state, train_ds, config, input_rng)
        _, _, test1_accuracy, _ = apply_model(state, test1_ds['image'], test1_ds['label'], jnp.arange(len(test1_ds['image'])), config.lambda_l2, config.lambda_core)
        _, _, test2_accuracy, _ = apply_model(state, test2_ds['image'], test2_ds['label'], jnp.arange(len(test2_ds['image'])), config.lambda_l2, config.lambda_core)

        # Calculate misclassification rates for the last epoch directly
        misclassification_rate_test1 = 1 - test1_accuracy
        misclassification_rate_test2 = 1 - test2_accuracy

        logging.info(
            f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy * 100:.2f}, '
            f'test1_accuracy: {test1_accuracy * 100:.2f}, '
            f'test2_accuracy: {test2_accuracy * 100:.2f}, '
            f'core_penalty: {core_penalty:.4f}'
        )

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_accuracy', test1_accuracy, epoch)
        summary_writer.scalar('test_accuracy', test2_accuracy, epoch)
        summary_writer.scalar('core_penalty', core_penalty, epoch)

    base_dir = os.path.dirname(workdir)
    misclassification_rates_file = os.path.join(base_dir, 'misclass_rates_models.txt')
    os.makedirs(base_dir, exist_ok=True)

    with open(misclassification_rates_file, 'a') as f:
        model_name = os.path.basename(workdir)
        f.write(f'Model: {model_name}, Test1 Misclassification Rate: {misclassification_rate_test1:.4f}, Test2 Misclassification Rate: {misclassification_rate_test2:.4f}\n')

    summary_writer.flush()
    return state
