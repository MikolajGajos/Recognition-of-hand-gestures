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
epochs = 20

train_val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=42
)
images_train_val = []
labels_train_val = []
for images, labels in train_val_dataset:

    images_train_val.append(images.numpy())
    labels_train_val.append(labels.numpy())

images_train_val = np.concatenate(images_train_val)
labels_train_val = np.concatenate(labels_train_val)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)
images_test = []
labels_test = []
for images, labels in test_dataset:
    images_test.append(images.numpy())
    labels_test.append(labels.numpy())

images_test = np.concatenate(images_test)
labels_test = np.concatenate(labels_test)

kf = KFold(n_splits=splits, shuffle=True, random_state=42)

max_accuracy = 0
train_acc = []
val_acc = []
epochs_range = []
folds_accuracy = []
for train_index, val_index in kf.split(images_train_val):

    images_train, images_val = images_train_val[train_index], images_train_val[val_index]
    labels_train, labels_val = labels_train_val[train_index], labels_train_val[val_index]

    model = Sequential([
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(train_val_dataset.class_names), name="outputs")])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(images_train, labels_train,
                        validation_data=(images_val, labels_val),
                        epochs=epochs)

    predictions = model.predict(images_test)
    predictions = np.argmax(predictions, axis=1)

    test_accuracy = metrics.accuracy_score(labels_test, predictions)
    folds_accuracy.append(test_accuracy)

    if test_accuracy > max_accuracy:
        max_accuracy = test_accuracy
        model.save("models/model_dataset1")

    epochs_range.append(range(epochs))
    train_acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])

for i in range(folds_accuracy.__len__()):
    print(f'Fold {i} test accuracy: {folds_accuracy[i]}')

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
