import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from keras import layers
from keras.models import Sequential

from sklearn import metrics
from sklearn.model_selection import KFold

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect-ds/Dataset1")

batch_size = 32
img_height = 180
img_width = 180
IMG_SIZE = (img_width, img_height)

splits = 5
epochs = 8

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=42
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names = train_dataset.class_names
num_classes = len(class_names)


def create_model():
    base_model = Sequential([
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])
    return base_model


X_test = []
y_test = []
for images, labels in test_dataset:
    X_test.append(images.numpy())
    y_test.append(labels.numpy())

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

X = []
y = []
for images, labels in train_dataset:
    X.append(images.numpy())
    y.append(labels.numpy())

X = np.concatenate(X)
y = np.concatenate(y)

predicted_y = []
expected_y = []

kf = KFold(n_splits=splits, shuffle=True, random_state=42)

fold = 0
max_accuracy = 0
train_acc = []
val_acc = []
epochs_range = []
for train_index, test_index in kf.split(X):
    fold += 1

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

    model = create_model()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs)

    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = metrics.accuracy_score(y_test, predicted_labels)
    expected_y.append(y_test)
    predicted_y.append(predicted_labels)
    print(f'Accuracy: {accuracy}')

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        model.save("models/model")

    epochs_range.append(range(epochs))
    train_acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])

expected_y = np.concatenate(expected_y)
predicted_y = np.concatenate(predicted_y)
accuracy = metrics.accuracy_score(expected_y, predicted_y)
print(f'Accuracy: {accuracy}')

fig, axs = plt.subplots(nrows=1, ncols=splits, layout="constrained", sharey=True, figsize=(11, 5))
i = 0
for ax in axs.flat:
    ax.plot(epochs_range[i], train_acc[i], label='Training Accuracy')
    ax.plot(epochs_range[i], val_acc[i], label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Fold {i}')
    ax.legend(loc='lower right')
    i += 1

plt.show()
