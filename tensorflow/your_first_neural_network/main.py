from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Download the fashion mnist database
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ============================================================================
#                                Pre-process the data
# ============================================================================
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Normalize the grayscale values to a 0.0-1.0 scale
# NOTE: Must do this for both train and test images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images from the training data and display the
# class name below each image
#plt.figure(figsize=(10, 10))
#for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

# ============================================================================
#                                Build the model
# ============================================================================


# -----------------------[ Setting up the layers ]----------------------------
# Most of deep learning consists of chaining together simple layers, most of
# which have parameters that are learned during training.
# The first layers.Flatten transforms the 28x28 images into a 1d array
# of 28x28 = 784 pixels.
# After the pixels are flattened, the network consists of a sequence of two
# layers.Dense layers. These are fully connected neural layers (which means
# that each node connects to every node of the next layer).
# The last layers.Dense layer is a softmax layer, which returns an array of
# probabilities that sum to 1. Each node contains a probability that the
# current image belongs to one of the 10 classes.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# --------------------------[ Compile the model ]----------------------------
# Loss Function - This measures how accurate the model is during training.
#                 We want to minimize this function to 'steer' the model
#                 in the right direction.
# Optimizer - This is how the model is updated based on the data it sees
#             and its loss function.
# Metrics - Used to monitor the training and testing steps. The following
#           example uses accuracy, the fraction of images that are
#           correctly classified.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------[ Train the model ]----------------------------
# Steps:
# 1. Feed training data into the model (train_images, train_labels)
# 2. Model 'learns' to associate images and labels
# 3. We ask the model to make predictions about a test set (test_images) array.
#    We verify that the predictions match the labels from the test_labels array
model.fit(train_images, train_labels, epochs=5)

# -----------------------------[ Evaluate accuracy ]---------------------------
# The accuracy on the test dataset is a little elss than the accuracy on the
# training dataset.  This gap between training and test accuracy is an example
# of overfitting. Overfitting is when a ML model performs worse on new data
# than on its training data.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

# -----------------------------[ Make predictions ]----------------------------
predictions = model.predict(test_images)


# -----------------------------[ Graph the predictions ]-----------------------
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'red'
    if predicted_label == true_label:
        color = 'green'

    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')


# Look at the 0th image, predictions, and predictions array
#i = 0
#plt.figure(figsize=(6, 3))
#plt.subplot(1, 2, 1)
#plot_image(i, predictions, test_labels, test_images)
#plt.subplot(1, 2, 2)
#plot_value_array(i, predictions, test_labels)
#plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# -----------------------------[ Make single prediction ]----------------------
img = test_images[0]

# tf.keras models are optimized to make predictions on a batch of examples.
# Even though we're using a single image, we need to add it to a list.
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
