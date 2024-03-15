"""
Author: Luka Secilmis

Description:
    Augments datasets with random simulated domain shifts.
"""
import tensorflow_datasets as tfds
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import jax.numpy as jnp
SEED = 7
np.random.seed(SEED)


def load_original_mnist():
    """Load the original MNIST dataset."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    # Normalize images
    train_ds['image'] = train_ds['image'] / 255.0
    test_ds['image'] = test_ds['image'] / 255.0
    
    return train_ds, test_ds


def train_mnist(images, labels, c, bidirec):
    """Augment training dataset with c rotated images."""
    images_aug = np.copy(images) 
    labels_aug = np.copy(labels)
    ids_aug = np.arange(images.shape[0])
    sign = 1 
    if bidirec:
        sign = np.random.choice([-1, 1], size=c)  # Randomly sample c directions to rotate by
    angles = np.random.uniform(35, 70, size=c)
    angles = angles * sign
    indices = np.random.choice(images.shape[0], size=c, replace=False)  # Uniformly sample c indices of images to rotate
    
    rotated_images = []
    for id, angle in zip(indices, angles):
        rot_img = scipy.ndimage.rotate(images[id], angle, reshape=False, mode='nearest')  # Rotate image by angle
        rotated_images.append(rot_img)
    
    images_aug = np.concatenate((images_aug, rotated_images), axis=0)
    labels_aug = np.concatenate((labels_aug, [labels[i] for i in indices]), axis=0)
    ids_aug = np.concatenate((ids_aug, indices), axis=0)
    
    return {'image': images_aug, 'label': labels_aug, 'id': ids_aug}


def test_mnist(images, bidirec):
    """Test set of only rotated images."""
    images_mod = np.copy(images) 

    # Rotate each image by a random angle
    for i in range(images.shape[0]):
        sign = 1 
        if bidirec: 
            sign = np.random.choice([-1, 1]) 
        angle = np.random.uniform(35, 70)
        images_mod[i] = scipy.ndimage.rotate(images[i], sign * angle, reshape=False, mode='nearest')

    return images_mod


def load_datasets_mnist(train_size, aug_size, bidirec):
    """Generate three datasets: augmented training set, test set with only rotated images, and original mnist test data."""
    train_ds, test_ds = load_original_mnist()

    # Subsample training set, and augment it
    indices = np.random.choice(train_ds['image'].shape[0], size=train_size, replace=False)
    train_ds = {'image': train_ds['image'][indices], 'label': train_ds['label'][indices]}
    train_ds_aug = train_mnist(train_ds['image'], train_ds['label'], aug_size, bidirec)

    # Test set with all images rotated
    test_ds_mod = {'image': test_mnist(test_ds['image'], bidirec), 'label': test_ds['label']}

    # Original test set
    test_ds = {'image': test_ds['image'], 'label': test_ds['label']}

    # Convert to jax arrays
    train_ds = {k: jnp.array(v) for k, v in train_ds.items()}
    test_ds_mod = {k: jnp.array(v) for k, v in test_ds_mod.items()}
    test_ds = {k: jnp.array(v) for k, v in test_ds.items()}

    return train_ds_aug, test_ds_mod, test_ds
    

def load_datasets(dataset, train_size, aug_size, bidirec=False):
    if dataset == 'mnist':
        return load_datasets_mnist(train_size, aug_size, bidirec)
    else: 
        raise ValueError('Dataset not supported yet.')


def show_img(img, ax=None, title=None):
  if ax is None:
    ax = plt.gca()
  ax.imshow(img[..., 0], cmap='gray')
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    ax.set_title(title)

def show_img_grid(imgs, titles):
  n = int(np.ceil(len(imgs)**.5))
  _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
  for i, (img, title) in enumerate(zip(imgs, titles)):
    show_img(img, axs[i // n][i % n], title)
