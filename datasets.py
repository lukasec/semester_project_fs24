import tensorflow_datasets as tfds
import jax.numpy as jnp


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.0  # Normalize pixels
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
    return train_ds, test_ds
