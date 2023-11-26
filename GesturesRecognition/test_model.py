import keras.models
import numpy as np
import os

import tensorflow as tf
from sklearn import metrics

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect-ds/Dataset2")

batch_size = 32
img_height = 180
img_width = 180
IMG_SIZE = (img_width, img_height)

splits = 5
epochs = 8

test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

X_test = []
y_test = []
for images, labels in test_dataset:
    X_test.append(images.numpy())
    y_test.append(labels.numpy())

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

model = keras.models.load_model("models/model")

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

accuracy = metrics.accuracy_score(y_test, predicted_labels)
print(f'Accuracy: {accuracy}')
