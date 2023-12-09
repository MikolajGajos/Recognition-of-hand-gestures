import keras.models
import numpy as np
import os
import pandas as pd
import tensorflow as tf

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect-ds/Dataset3")

img_height = 180
img_width = 180
batch_size = 32

test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=42
)
class_names = test_dataset.class_names

images_test = []
labels_test = []
for images, labels in test_dataset:
    images_test.append(images.numpy())
    labels_test.append(labels.numpy())

images_test = np.concatenate(images_test)
labels_test = np.concatenate(labels_test)

model = keras.models.load_model("models/model_softmax")

data = np.array([['', 'False/True', 'mu', 'valfa', 'c', 'fish', 'star', 'E', 'W', 'spir', 'T', 'P', 'fing', 'L']])

for i in range(images_test.__len__()):
    image_batch = tf.expand_dims(images_test[i], 0)
    prediction = model.predict(image_batch)
    image_type = 'T'
    if labels_test[i] == 12:
        image_type = 'F'

    prediction = prediction.flatten()
    prediction = [format(num, '.4f') for num in prediction]
    new_row = np.array([i, image_type])
    new_row = np.concatenate((new_row, prediction))
    data = np.append(data, [new_row], axis=0)

data = pd.DataFrame(data=data[1:, 1:],
                    index=data[1:, 0],
                    columns=data[0, 1:])

data.to_excel("results.xlsx")

