import datetime
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver(url=os.getenv("KF_MONGODB_URL"), db_name=os.getenv("KF_SACRED_ID")))


class LogPerformance(tf.keras.callbacks.Callback):
    def on_epoch_end(self, _, logs={}):
        log_performance(logs=logs)


@ex.capture
def log_performance(_run, logs):
    _run.log_scalar("accuracy", float(logs.get('accuracy')))
    _run.result = float(logs.get('accuracy'))


@ex.main
def run():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x_train,
              y=y_train,
              epochs=100,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback, LogPerformance()])


if __name__ == "__main__":
    ex.run()
