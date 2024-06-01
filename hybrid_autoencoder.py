import os
from typing import Sequence
import jax
import jax.numpy as jnp
import numpy as np
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
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=256, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=512, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        z = nn.Dense(features=10)(x)
        return z


class CNNDecoder(nn.Module):
    """Convolutional Decoder."""
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(features=512 * 2 * 2)(z)
        x = x.reshape((x.shape[0], 2, 2, 512))

        x = nn.ConvTranspose(features=256, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)

        x_hat = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2))(x)
        x_hat = nn.sigmoid(x_hat)
        return x_hat
    

class AutoEncoder(nn.Module):
    def setup(self):
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    

@jax.jit
def apply_model(state, images, gender_anchor, gender_pos, gender_neg, id_anchor, id_pos, id_neg, lambda_reconstruction, lambda_gender, lambda_id, margin):
    def triplet_loss(anchor, positive, negative, margin=margin):
        d_pos = jnp.sum((anchor - positive) ** 2, axis=1)
        d_neg = jnp.sum((anchor - negative) ** 2, axis=1)
        loss = jnp.maximum(0.0, margin + d_pos - d_neg)
        return jnp.mean(loss)

    def loss_fn(params):
        # Reconstruction loss
        x_hat = state.apply_fn({'params': params}, images)
        reconstruction_loss = jnp.mean((images - x_hat) ** 2)

        # Gender triplet loss
        gender_z = state.apply_fn({'params': params}, gender_anchor, method=AutoEncoder().encode)
        gender_z_pos = state.apply_fn({'params': params}, gender_pos, method=AutoEncoder().encode)
        gender_z_neg = state.apply_fn({'params': params}, gender_neg, method=AutoEncoder().encode)
        gender_loss = triplet_loss(gender_z, gender_z_pos, gender_z_neg)

        # ID triplet loss
        id_z = state.apply_fn({'params': params}, id_anchor, method=AutoEncoder().encode)
        id_z_pos = state.apply_fn({'params': params}, id_pos, method=AutoEncoder().encode)
        id_z_neg = state.apply_fn({'params': params}, id_neg, method=AutoEncoder().encode)
        id_loss = triplet_loss(id_z, id_z_pos, id_z_neg, margin=100.0)

        # Reconstruction + contrastive loss
        loss = lambda_reconstruction * reconstruction_loss + lambda_gender * gender_loss + lambda_id * id_loss
        return loss, (reconstruction_loss, gender_loss, id_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (reconstruction_loss, gender_loss, id_loss)), grads = grad_fn(state.params)
    return grads, loss, reconstruction_loss, gender_loss, id_loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def select_triplets_gender(images, labels):
    """Select triplet samples from batch."""
    men = images[labels == 0]
    women = images[labels == 1]

    # Balanced triplet selection
    n = min(len(men), len(women)) // 3

    men_anchors = men[:n]
    men_positives = men[n:2*n]
    men_negatives = men[2*n:3*n]

    women_anchors = women[:n]
    women_positives = women[n:2*n]
    women_negatives = women[2*n:3*n]

    anchors = jnp.concatenate([men_anchors, women_anchors], axis=0)
    positives = jnp.concatenate([men_positives, women_positives], axis=0)  # Cross-gender positive pairing
    negatives = jnp.concatenate([women_negatives, men_negatives], axis=0)  # Cross-gender negative pairing
    return anchors, positives, negatives


def select_triplets_id(images, ids, labels, rng):
    def sub_select_triplets(images, ids, rng):
        if len(images) < 3:
            return jnp.array([]), jnp.array([]), jnp.array([])
        else:
            unique_ids, _, counts = jnp.unique(ids, return_index=True, return_counts=True)
            
            # Number of anchors = number of unique IDs that occur more than once
            mask = counts > 1
            valid_ids = unique_ids[mask]
            
            if len(valid_ids) == 0:
                return jnp.array([]), jnp.array([]), jnp.array([])

            anchor_idx = []
            pos_idx = []
            neg_idx = []

            for valid_id in valid_ids:
                rng, neg_rng = jax.random.split(rng, 2)
                
                id_idx = jnp.where(ids == valid_id)[0]
                
                anchor_idx.append(id_idx[0])
                pos_idx.append(id_idx[1])
                
                neg_idx_possible = jnp.where(ids != valid_id)[0]
                
                chosen_neg_idx = jax.random.choice(neg_rng, neg_idx_possible, shape=())
                neg_idx.append(chosen_neg_idx)

            anchors = images[jnp.array(anchor_idx)]
            positives = images[jnp.array(pos_idx)]
            negatives = images[jnp.array(neg_idx)]

            return anchors, positives, negatives

    men = images[labels == 0]
    men_ids = ids[labels == 0]
    women = images[labels == 1]
    women_ids = ids[labels == 1]
    men_anchors, men_positives, men_negatives = sub_select_triplets(men, men_ids, rng)
    women_anchors, women_positives, women_negatives = sub_select_triplets(women, women_ids, rng)

    # concatenate only if there are triplets
    if (len(women_anchors) > 0) and (len(men_anchors) > 0):
        anchors = jnp.concatenate([men_anchors, women_anchors], axis=0)
        positives = jnp.concatenate([men_positives, women_positives], axis=0)  
        negatives = jnp.concatenate([men_negatives, women_negatives], axis=0) 
        return anchors, positives, negatives
    elif len(women_anchors) > 0:
        return women_anchors, women_positives, women_negatives
    else:
        return men_positives, men_positives, men_negatives


def train_epoch(state, train_ds, config, rng):
    """Train for one epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // config.batch_size
    
    # Shuffle dataset at start of each epoch
    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[: steps_per_epoch * config.batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    epoch_loss = []
    epoch_gender_len = []
    epoch_id_len = []
    epoch_reconstruction_loss = []
    epoch_gender_loss = []
    epoch_id_loss = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        batch_ids = train_ds['id'][perm, ...]

        batch_gender_anchor, batch_gender_pos, batch_gender_neg = select_triplets_gender(batch_images, batch_labels)
        batch_id_anchor, batch_id_pos, batch_id_neg = select_triplets_id(batch_images, batch_ids, batch_labels, rng)
        epoch_gender_len.append(len(batch_gender_anchor))
        epoch_id_len.append(len(batch_id_anchor))

        grads, loss, reconstruction_loss, gender_loss, id_loss = apply_model(state, batch_images, batch_gender_anchor, batch_gender_pos, batch_gender_neg, batch_id_anchor, batch_id_pos, batch_id_neg, config.lambda_reconstruction, config.lambda_gender, config.lambda_id, config.margin)
        state = update_model(state, grads)
        
        epoch_loss.append(loss)
        epoch_reconstruction_loss.append(reconstruction_loss)
        epoch_gender_loss.append(gender_loss)
        epoch_id_loss.append(id_loss)
        

    train_loss = np.mean(epoch_loss)
    gen_len = np.mean(epoch_gender_len)
    id_len = np.mean(epoch_id_len)
    reconstruction_loss = np.mean(epoch_reconstruction_loss)
    gender_loss = np.mean(epoch_gender_loss)
    id_loss = np.mean(epoch_id_loss)

    return state, train_loss, gen_len, id_len, reconstruction_loss, gender_loss, id_loss


def create_train_state(rng, config):
    """Initialize weights and optimizer."""
    model = AutoEncoder()
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
        rng, _ = jax.random.split(rng)
        state, train_loss, gen_len, id_len, reconstruction_loss, gender_loss, id_loss = train_epoch(state, train_ds, config, rng)
        
        # Log results
        logging.info(
            f'epoch: {epoch}, train_loss: {train_loss:.4f}, '
            f'reconstruction loss: {reconstruction_loss:.4f}, '
            f'gender loss: {gender_loss:.4f}, '
            f'id loss: {id_loss:.4f}, '
            f'gender anchors: {gen_len:.4f}, '
            f'id anchors: {id_len:.4f}'
            )
        
        summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.flush()
    return state
