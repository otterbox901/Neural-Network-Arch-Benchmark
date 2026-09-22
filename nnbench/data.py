"""Real-world datasets and a strict train / validation / test split.

The official test split of each dataset is held out as the *independent* test
set: it is never used for fitting, normalisation statistics, early stopping or
model selection. Validation data is carved (stratified) out of the official
training split and is the only data used for training-time decisions.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLASS_NAMES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "fashion_mnist": [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot",
    ],
    "mnist": [str(i) for i in range(10)],
}

_LOADERS = {
    "cifar10": tf.keras.datasets.cifar10.load_data,
    "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
    "mnist": tf.keras.datasets.mnist.load_data,
}


@dataclass
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list

    @property
    def input_shape(self):
        return self.x_train.shape[1:]

    @property
    def num_classes(self):
        return len(self.class_names)


def load_splits(name, val_fraction=0.1, seed=0, train_subset=None, test_subset=None):
    """Download (cached in ~/.keras/datasets) and split a dataset.

    ``train_subset`` / ``test_subset`` limit sample counts for quick smoke runs.
    """
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(_LOADERS)}")

    (x_full, y_full), (x_test, y_test) = _LOADERS[name]()
    y_full, y_test = y_full.reshape(-1), y_test.reshape(-1)
    if x_full.ndim == 3:  # grayscale -> add channel axis
        x_full, x_test = x_full[..., None], x_test[..., None]

    if train_subset:
        x_full, _, y_full, _ = train_test_split(
            x_full, y_full, train_size=train_subset, stratify=y_full, random_state=seed)
    if test_subset:
        x_test, _, y_test, _ = train_test_split(
            x_test, y_test, train_size=test_subset, stratify=y_test, random_state=seed)

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=val_fraction, stratify=y_full, random_state=seed)

    # Normalisation statistics come from the training portion only (no leakage).
    x_train = x_train.astype("float32") / 255.0
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True) + 1e-7

    def norm(x):
        return ((x.astype("float32") / 255.0) - mean) / std

    return Splits(
        x_train=(x_train - mean) / std, y_train=y_train,
        x_val=norm(x_val), y_val=y_val,
        x_test=norm(x_test), y_test=y_test,
        class_names=CLASS_NAMES[name],
    )


def make_dataset(x, y, batch_size, shuffle=False, augment=False, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
        ])
        ds = ds.map(lambda a, b: (aug(a, training=True), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
