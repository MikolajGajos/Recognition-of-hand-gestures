import keras.models
import numpy as np
import pandas as pd
import os
import sklearn.metrics
import tensorflow as tf

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect-ds/Dataset1")

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

model_name = 'model_softmax'
model = keras.models.load_model(f"models/{model_name}")

names = ['mu', 'valfa', 'c', 'fish', 'star', 'E', 'W', 'spir', 'T', 'P', 'fing', 'h']

predictions = model.predict(images_test)
predictions = np.argmax(predictions, axis=1)
confusion_matrix = sklearn.metrics.confusion_matrix(y_true=labels_test, y_pred=predictions)

data_frame = pd.DataFrame(confusion_matrix, columns=names, index=names)
print(data_frame)
data_frame.to_excel(f'excel_results/{model_name}_confusion_matrix.xlsx')

