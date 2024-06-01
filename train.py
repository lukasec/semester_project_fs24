"""
Author: Luka Secilmis

Description:
    Implements Conditional Variance Regularization (CoRe) for classification.
"""
import os
from absl import logging
from flax.metrics import tensorboard
from flax.training import train_state
from flax.training.early_stopping import EarlyStopping
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import data_handler
import architectures
import hybrid_autoencoder
logging.set_verbosity(logging.INFO)
SEED = 7


def apply_model(state, images, labels, ids, num_classes, lambda_l2, lambda_core, cfl_anneal):
    """Computes gradients, loss and accuracy for a single batch."""
    def core_penalty(params, logits, ids):
        # Note: for our classification experiments, it suffices to condition on ID (as Y is always the same for a given an ID)
        # if we wish to condition on (ID, Y) we simply create a composite key ID := (ID,Y) when processing datasets
        key = ids
        _, index_inverse, counts = jnp.unique(key, return_inverse=True, return_counts=True)
        mask = counts[index_inverse] > 1
        penalty = 0.0
        
        if jnp.any(mask):
            # IDs with more than one occurence
            filtered_logits = logits[mask] 
            
            # Mean logits per ID
            sum_perID = jax.ops.segment_sum(filtered_logits, index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            count_perID = jax.ops.segment_sum(jnp.ones_like(filtered_logits[:, 0]), index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            mean_perID = sum_perID / count_perID[:, None]
            
            # Variance per ID
            square_diff = (filtered_logits - mean_perID[index_inverse[mask]])**2
            sum_squares_perID = jax.ops.segment_sum(square_diff, index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            variance_perID = sum_squares_perID / count_perID[:, None]
            
            # Mean variance
            penalty = jnp.mean(variance_perID[count_perID > 1])
        return penalty
        
    def l2_penalty(params):
        return sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params))
 
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)  # forward pass
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        l2_pen = l2_penalty(params)  # Ridge penalty
        loss += lambda_l2 * l2_pen  
        core_pen = core_penalty(params, logits, ids)  # CoRe penalty
        loss += cfl_anneal * lambda_core * core_pen  # potentially annealed
        return loss, (logits, core_pen)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # auxiliary data
    (loss, (logits, core_pen)), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy, core_pen


def apply_regression_model(state, images, labels, ids, lambda_l2, lambda_core, cfl_anneal):
    """Regression: computes gradients, loss and RMSE for a single batch."""
    def core_penalty(params, logits, ids):
        key = ids
        _, index_inverse, counts = jnp.unique(key, return_inverse=True, return_counts=True)
        mask = counts[index_inverse] > 1
        penalty = 0.0
        
        if jnp.any(mask):
            # IDs with more than one occurence
            filtered_logits = logits[mask] 
            
            # Mean logits per ID
            sum_perID = jax.ops.segment_sum(filtered_logits, index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            count_perID = jax.ops.segment_sum(jnp.ones_like(filtered_logits[:, 0]), index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            mean_perID = sum_perID / count_perID[:, None]
            
            # Variance per ID
            square_diff = (filtered_logits - mean_perID[index_inverse[mask]])**2
            sum_squares_perID = jax.ops.segment_sum(square_diff, index_inverse[mask], num_segments=jnp.max(index_inverse) + 1)
            variance_perID = sum_squares_perID / count_perID[:, None]
            
            # Mean variance
            penalty = jnp.mean(variance_perID[count_perID > 1])
        return penalty
        
    def l2_penalty(params):
        return sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params))
 
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, images)  # forward pass
        loss = jnp.mean((predictions - labels)**2)  # MSE 
        l2_pen = l2_penalty(params)  # Ridge penalty
        loss += lambda_l2 * l2_pen  
        core_pen = core_penalty(params, predictions, ids)  # CoRe penalty
        loss += cfl_anneal * lambda_core * core_pen  # potentially annealed
        return loss, (predictions, core_pen)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # auxiliary data
    (loss, (_, core_pen)), grads = grad_fn(state.params)
    return grads, loss, core_pen


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, config, rng, cfl_anneal):
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
    ids_more_than_once = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        batch_ids = train_ds['id'][perm, ...]

        if config.model != 'regression':
            grads, loss, accuracy, core_pen = apply_model(state, batch_images, batch_labels, batch_ids, config.num_classes, config.lambda_l2, config.lambda_core, cfl_anneal)
        else:
            grads, loss, core_pen = apply_regression_model(state, batch_images, batch_labels, batch_ids, config.lambda_l2, config.lambda_core, cfl_anneal)
            accuracy = 0  # not needed for regression
        
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
        core_penalties.append(core_pen)

        # count number of batch IDs that occur more than once (contributing to CoRe penalty)
        count_id = jnp.bincount(batch_ids)
        count_id = count_id[count_id > 1]
        ids_more_than_once.append(jnp.sum(count_id))

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    core_penalty= np.mean(core_penalties)
    ids_more_than_once = np.mean(ids_more_than_once)
    return state, train_loss, train_accuracy, core_penalty, ids_more_than_once


def create_train_state(rng, config):
    """Initialize weights and optimizer."""
    # Model
    if config.model == 'mnist':
        model = architectures.CNN_mnist()
        params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']

    elif config.model == 'celebA':
        model = architectures.CNN_celebA()
        params = model.init(rng, jnp.ones([1, 64, 64, 3]))['params']

    elif config.model == 'synthetic':
        model = architectures.MLP()
        params = model.init(rng, jnp.ones([1, 2]))['params']  
    
    elif config.model == 'regression':
        model = hybrid_autoencoder.CNNEncoder()
        params = model.init(rng, jnp.ones([1, 64, 64, 3]))['params']

    # Optimizer
    if config.schedule == 'exp_decay':
        schedule = optax.exponential_decay(init_value=config.learning_rate,
                                            transition_steps=config.decay_steps,
                                            decay_rate=config.decay_rate)

    elif config.schedule == 'warmup_cosine_decay':
        schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                        peak_value=config.learning_rate,
                                                        warmup_steps=config.warmup_steps,
                                                        decay_steps=config.decay_steps,
                                                        end_value=config.end_learning_rate) 
    
    elif config.schedule == 'warmup_exp_decay':
        schedule = optax.warmup_exponential_decay_schedule(init_value=0.0,
                                                           peak_value=config.learning_rate,
                                                           warmup_steps=config.warmup_steps,
                                                           transition_steps=config.decay_steps,
                                                           decay_rate=config.decay_rate)
    
    else: schedule = config.learning_rate  # constant lr
    tx = optax.adam(schedule)    
   
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, train_ds: dict, test1_ds: dict, test2_ds: dict) -> train_state.TrainState:
    rng = jax.random.key(SEED)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    # Set up validation and early stopping
    val_loss, val_loss_str = -1.0, "N/A"
    if config.with_val or config.with_earlystop: train_ds, val_ds = data_handler.split_train_validation(train_ds, config.val_size)
    if config.with_earlystop: early_stop = EarlyStopping(min_delta=config.delta, patience=config.patience)

    # Seat up annealing of CoRe penalty
    if config.cfl_anneal: cfl_anneal, cfl_rise = 0.0, 1.0 / ((1 - config.no_cfl_frac) * config.num_epochs)
    elif config.no_cfl_frac is not None: cfl_anneal, cfl_rise = 0.0, 0.0
    else: cfl_anneal, cfl_rise = 1.0, 0.0

    for epoch in range(1, config.num_epochs + 1):
        # Train for one epoch
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy, core_penalty, ids_more_than_once = train_epoch(state, train_ds, config, input_rng, cfl_anneal)

        # Evaluate on validation set
        if config.with_val or config.with_earlystop:
            _, val_loss, _, _ = apply_model(state, val_ds['image'], val_ds['label'], val_ds['id'], config.num_classes, config.lambda_l2, config.lambda_core, cfl_anneal)
            val_loss_str = "{val_loss:.4f}"

        # Early stopping
        if config.with_earlystop: early_stop = early_stop.update(val_loss)

        # Evaluate on test sets & Llog results
        if config.model != 'regression':  # classification
            _, test1_loss, test1_accuracy, _ = apply_model(state, test1_ds['image'], test1_ds['label'], jnp.arange(len(test1_ds['image'])), config.num_classes, config.lambda_l2, config.lambda_core, cfl_anneal)
            _, test2_loss, test2_accuracy, _ = apply_model(state, test2_ds['image'], test2_ds['label'], jnp.arange(len(test2_ds['image'])), config.num_classes, config.lambda_l2, config.lambda_core, cfl_anneal)

            logging.info(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy * 100:.2f}, '
                f'test1_accuracy: {test1_accuracy * 100:.2f}, '
                f'test2_accuracy: {test2_accuracy * 100:.2f}, '
                f'core_penalty: {core_penalty:.4f}, '
                f'val_loss: {val_loss_str}, '
                f'ids_more_than_once: {ids_more_than_once:.2f}' 
                )
            summary_writer.scalar('train_loss', train_loss, epoch)
            summary_writer.scalar('train_accuracy', train_accuracy, epoch)
            summary_writer.scalar('test1_accuracy', test1_accuracy, epoch)
            summary_writer.scalar('test2_accuracy', test2_accuracy, epoch)
            summary_writer.scalar('core_penalty', core_penalty, epoch)
            summary_writer.scalar('val_loss', val_loss, epoch)
        else:  # regression
            _, test1_loss, _ = apply_regression_model(state, test1_ds['image'], test1_ds['label'], jnp.arange(len(test1_ds['image'])), config.lambda_l2, config.lambda_core, cfl_anneal)
            _, test2_loss, _ = apply_regression_model(state, test2_ds['image'], test2_ds['label'], jnp.arange(len(test2_ds['image'])), config.lambda_l2, config.lambda_core, cfl_anneal)

            logging.info(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, '
                f'test1_loss: {test1_loss:.4f}, '
                f'test2_loss: {test2_loss:.4f}, '
                f'core_penalty: {core_penalty:.4f}, '
                f'val_loss: {val_loss_str}, '
                f'ids_more_than_once: {ids_more_than_once:.2f}'
            )
            summary_writer.scalar('train_loss', train_loss, epoch)
            summary_writer.scalar('test1_MSE', test1_loss, epoch)
            summary_writer.scalar('test2_MSE', test2_loss, epoch)
            summary_writer.scalar('core_penalty', core_penalty, epoch)
            summary_writer.scalar('val_loss', val_loss, epoch)

        # Early stopping
        if config.with_earlystop and early_stop.should_stop:
            logging.info(f'Met early stopping criteria, breaking at epoch {epoch}.')
            break

        # Anneal CoRe penalty
        if config.cfl_anneal:  # core penalty annealing
            if epoch > int(config.no_cfl_frac * config.num_epochs): cfl_anneal += cfl_rise
        elif config.no_cfl_frac is not None:  # no annealing, but only use CoRe penalty after a certain fraction of epochs
            if epoch >= int(config.no_cfl_frac * config.num_epochs): cfl_anneal = 1.0
    
    summary_writer.flush()
    return state
