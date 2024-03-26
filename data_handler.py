"""
Author: Luka Secilmis

Description:
    Augments datasets with random simulated domain shifts.
"""
import os
import tensorflow_datasets as tfds
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
SEED = 7
np.random.seed(SEED)


########################### Experiment 5.5 MNIST with rotations ###########################
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


########################### Experiment 5.2 CelebA with confounding ###########################
def load_original_celebA():
    celebA_path = './data/celebA'
    if not os.path.exists(celebA_path):  # Requires manual download
        raise FileNotFoundError('CelebA dataset not found.\nManually download from Kaggle and place in data/celebA.\nhttps://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download \nAlso: unzip img_align_celeba.zip in Img subfolder.\nAlso: on the second line of list_attr_celeba.txt there is a column name missing. Please add the first column to be image_id.')

    identity = pd.read_csv(os.path.join(celebA_path, 'Anno/identity_CelebA.txt'), sep='\s+', header=None, names=['image_id', 'ID'])
    attributes = pd.read_csv(os.path.join(celebA_path, 'Anno/list_attr_celeba.txt'), sep='\s+', header=1)[['image_id', 'Male', 'Eyeglasses']]
    attributes = attributes.replace(-1, 0)
    partition = pd.read_csv(os.path.join(celebA_path, 'Eval/list_eval_partition.txt'), sep='\s+', header=None, names=['image_id', 'partition'])
    img_dir = os.path.join(celebA_path, 'Img/img_align_celeba')

    train_ds = partition[partition['partition'] == 0].merge(identity, on='image_id').merge(attributes, on='image_id')
    validation_ds = partition[partition['partition'] == 1].merge(identity, on='image_id').merge(attributes, on='image_id')
    test_ds = partition[partition['partition'] == 2].merge(identity, on='image_id').merge(attributes, on='image_id')
    return train_ds, validation_ds, test_ds, img_dir


def load_datasets_celebA_counfound(prop_glasses, train_size, test_size):
    """Create training and test sets with confounding as specified in section 5.2."""
    train_ds, validation_ds, test_ds, img_dir = load_original_celebA()
    
    # Combine all datasets and shuffle
    all_ds = pd.concat([train_ds, validation_ds, test_ds])
    all_ds = all_ds.sample(frac=1, random_state=SEED).reset_index(drop=True)  # Permute dataset

    # Craft subsets and sample from them
    n = train_size // 2  # Train: half men, half women
    m = int(n * prop_glasses)  # Train: nb of men with glasses, women without
    t1 = test_size[0] // 2  # Test set 1: half men with glasses, half women without
    t2 = test_size[1] // 2  # test set 2: vice versa
    men_with_glasses = all_ds[(all_ds['Male'] == 1) & (all_ds['Eyeglasses'] == 1)].sample(n=m+t1 , random_state=SEED)
    men_without_glasses = all_ds[(all_ds['Male'] == 1) & (all_ds['Eyeglasses'] == 0)].sample(n=n-m+t2, random_state=SEED)
    women_with_glasses = all_ds[(all_ds['Male'] == 0) & (all_ds['Eyeglasses'] == 1)].sample(n=n-m+t2, random_state=SEED)
    women_without_glasses = all_ds[(all_ds['Male'] == 0) & (all_ds['Eyeglasses'] == 0)].sample(n=m+t1, random_state=SEED)

    # Construct datasets with confounding
    # Training set: men mostly wearing glasses, women mostly not wearing glasses
    train_ds = pd.concat([men_with_glasses[:m], men_without_glasses[:n-m], women_with_glasses[:n-m], women_without_glasses[:m]])
    train_ds = train_ds.sample(frac=1, random_state=SEED).reset_index(drop=True)  

    # Test set 1: men only wear glasses, women all without glasses
    test1_ds = pd.concat([men_with_glasses[m:m+t1], women_without_glasses[m:m+t1]])
    test1_ds = test1_ds.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Test set 2: women only wear glasses, men all without glasses
    test2_ds = pd.concat([men_without_glasses[n-m:n-m+t2], women_with_glasses[n-m:n-m+t2]])
    test2_ds = test2_ds.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return train_ds, test1_ds, test2_ds, img_dir
        

def load_and_preprocess_image(img_path, output_shape=(64, 64)):
    """Load an image and resize it to desired output shape."""
    img = imread(img_path)
    img_resized = resize(img, output_shape, anti_aliasing=True)
    return img_resized


def conv_celebA_to_jax(df, img_dir):
    """Convert CelebA dataset to JAX format."""
    images = []

    for _, row in df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(img_dir, img_id)
        img = load_and_preprocess_image(img_path, output_shape=(64, 64, 3))
        images.append(np.array(img))
    images = np.array(images)
    labels = np.array(df['Male'].values)  # Ensure labels are integers
    ids = np.array(df['ID'].values) if 'ID' in df.columns else None

    if ids is not None:
        ds = {'image': images, 'label': labels, 'id': ids}
    else:
        ds = {'image': images, 'label': labels}
    # return {k: jnp.array(v) for k, v in ds.items()}
    return ds


########################### Main function ###########################
def load_datasets(dataset, train_size, aug_size, bidirec=False):
    if dataset == 'mnist':
        return load_datasets_mnist(train_size, aug_size, bidirec)
    elif dataset == 'celebA_confound':
        return load_datasets_celebA_counfound()
    else: 
        raise ValueError('Dataset not supported yet.')
