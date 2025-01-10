#Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Importin dataset
datagen = ImageDataGenerator(rotation_range = 40,
                             with_shift_range = 0.2,
                             height_shift_range = 0.2,
                             rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             fill_mode = 'nerest')
trained_datagen = datagen.flow_from_directory('XXXXXXXXX',
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'categorical')

#Creating model
cnn = keras.models.sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))