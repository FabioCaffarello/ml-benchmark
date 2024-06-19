import time
import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical


logger = logging.getLogger(__name__)


def load_data():
    data = np.load('/app/data/cifar10_train.npz')
    x_train, y_train = data['x'], data['y']
    data = np.load('/app/data/cifar10_test.npz')
    x_test, y_test = data['x'], data['y']
    return x_train, y_train, x_test, y_test


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def benchmark():
    x_train, y_train, x_test, y_test = load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = create_model()

    start_time = time.time()
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    end_time = time.time()

    logger.info(f"Training completed in {end_time - start_time} seconds")


if __name__ == "__main__":
    benchmark()
