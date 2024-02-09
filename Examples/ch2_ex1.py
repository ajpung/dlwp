import os
from keras.datasets import mnist
from keras import models, layers

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

(train_images,train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(len(train_labels))

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

