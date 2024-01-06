'''
Practicing with the MNIST dataset using the examples from 
Deep Learning with Python by Francois Chollet, Chapter 2
'''

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# The network architecture
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# The compilation step
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Preparing the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)

print(f"\narray({predictions[0]})")

print(f"\nargmax: {predictions[0].argmax()}")

print(f"\npredictions[0][7]: {predictions[0][7]}")

print(f"\ntests_labels[0]: {test_labels[0]}")

# Evaluating the model on new data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\ntest_acc: {test_acc}")