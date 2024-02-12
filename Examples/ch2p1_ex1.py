import os
from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

(train_images,train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(len(train_labels))

# Define network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Reshape arrays into expected dimensions
train_images = train_images.reshape(60000,28*28)
train_images = train_images.astype('float32')/255
test_images = test_images.reshape(10000,28*28)
test_images = test_images.astype('float32')/255

# Categorically encode lables
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Fit the model to the data
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
