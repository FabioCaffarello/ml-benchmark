import numpy as np
import logging
from tensorflow.keras.datasets import cifar10


logger = logging.getLogger(__name__)


def _download_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    logger.info("CIFAR-10 dataset downloaded")
    return x_train, y_train, x_test, y_test


def _preprocess_cifar10(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    logger.info("CIFAR-10 dataset preprocessed")

    # Save dataset to npz files
    logger.info("Saving CIFAR-10 dataset to data/cifar10_train.npz and data/cifar10_test.npz")
    np.savez_compressed('data/cifar10_train.npz', x=x_train, y=y_train)
    np.savez_compressed('data/cifar10_test.npz', x=x_test, y=y_test)
    logger.info("CIFAR-10 dataset saved to data/cifar10_train.npz and data/cifar10_test.npz")


def download_and_preprocess():
    logger.info("Downloading CIFAR-10 dataset...")
    x_train, y_train, x_test, y_test = _download_cifar10()
    logger.info("Preprocessing CIFAR-10 dataset...")
    _preprocess_cifar10(x_train, y_train, x_test, y_test)
    logger.info("CIFAR-10 dataset downloaded and preprocessed")
