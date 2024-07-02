# Chest X-Ray Pneumonia Detection

This project aims to develop a deep learning model to detect pneumonia from chest X-ray images. Using convolutional neural networks (CNNs), we train and evaluate a model to classify X-ray images into two categories: pneumonia or normal.

## Project Setup

### 1. Set-up

To start, we perform the necessary imports:

```python
import os
import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import add
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

np.random.seed(777)
tf.random.set_seed(777)
```

### 2. Define Constants
One of the best practices when doing a machine learning project is to define constants together, thus facilitating further changes. Given that, we define the batch size, the height and width of the images, and the learning rate.

```python
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
LEARNING_RATE = 0.001
```

### 3. Data Preprocessing
Data augmentation is applied to artificially increase the size of the training dataset by creating modified versions of images in the dataset. This helps improve the robustness and performance of the model. We use ImageDataGenerator for this purpose

```python
# Data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    brightness_range=[0.5, 1.5]
)

test_datagen = ImageDataGenerator(rescale=1./255)
```
### 4. Model Architecture
The model used for this project is a convolutional neural network (CNN) designed to process and classify images. The architecture of the model includes various layers such as convolutional layers, pooling layers, and dense layers.

```python
# Define the CNN model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```
### 5. Training the Model
We use the training data to train the model with appropriate callbacks to save the best model and reduce the learning rate on plateau.

```python
# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint, reduce_lr]
)
```
### 6. Evaluating the Model
After training, we evaluate the model using test data and visualize the results using classification reports and confusion matrices.

```python
Copy code
# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Classification report and confusion matrix
y_pred = model.predict(test_generator)
y_pred = np.round(y_pred).astype(int)
print(classification_report(test_generator.classes, y_pred))
conf_matrix = confusion_matrix(test_generator.classes, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```
