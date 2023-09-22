import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

path = "C:/Users/Priyanka/AppData/Local/Programs/Python/Python311/Lib/site-packages/keras/datasets/mnist.npz"
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data(path=path)
print(len(X_train))
print(len(X_test))
print(X_train[0].shape)
plt.matshow(X_train[0])
plt.show()

print(y_train[0])

#scaling the input training data set -- to have values between 0 and 1
# Note: scaling improves the performance of the model

X_train = X_train/255
X_test = X_test/255

# flattening the input X_test and X_train as the neural network take one-dimensional array as the input
X_train_flattened = X_train.reshape(len(X_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)

#without hidden layer
#note: use relu for the hidden layers as it is more efficient that other activators
#use sigmoid for the output layer

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
