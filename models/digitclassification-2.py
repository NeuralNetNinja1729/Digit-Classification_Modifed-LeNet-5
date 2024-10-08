import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import json

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode='constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode='constant')

x_train.shape

nx, hx, wx = x_train.shape
ny, hy, wy = x_test.shape

x_train = x_train.reshape((nx, 32, 32, 1))
x_test = x_test.reshape((ny, 32, 32, 1))

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=9)

cnn1 = models.Sequential()
cnn1.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)))
cnn1.add(layers.BatchNormalization())
cnn1.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
# cnn1.add(layers.Dropout(0.25))

cnn1.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
cnn1.add(layers.BatchNormalization())
cnn1.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
# cnn1.add(layers.Dropout(0.25))

cnn1.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
cnn1.add(layers.BatchNormalization())
cnn1.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))

cnn1.add(layers.Flatten())

cnn1.add(layers.Dense(120, activation='relu'))
cnn1.add(layers.BatchNormalization())
cnn1.add(layers.Dropout(0.5))

cnn1.add(layers.Dense(84, activation='relu'))
cnn1.add(layers.BatchNormalization())
cnn1.add(layers.Dropout(0.5))

cnn1.add(layers.Dense(10, activation='softmax'))

cnn1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

cnn1.summary()

training_history = cnn1.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

test_loss, test_accuracy = cnn1.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

cnn1.save("digit_classification_cnn_with_BatchNormalizationAndDropout_2.h5")

with open('model_history.json', 'w') as f:
    json.dump(training_history.history, f)

