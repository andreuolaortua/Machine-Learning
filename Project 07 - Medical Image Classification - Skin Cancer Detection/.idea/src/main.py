import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

tf.__version__

#Prepocessing trainingset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('Skin cancer ISIC The International Skin Imaging Collaboration/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

#Prepocessing testset
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Skin cancer ISIC The International Skin Imaging Collaboration/Test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

#Initialising the CNN
cnn = tf.keras.models.Sequential()
#Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Flattening
cnn.add(tf.keras.layers.Flatten())
#Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
#Output layer
cnn.add(tf.keras.layers.Dense(units=9, activation='softmax'))

#Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

cnn.save('modelo_cancer_piel.h5')

img_paths = [
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000002.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000004.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000013.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000022.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000026.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000029.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000030.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000031.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000035.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000036.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000040.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000043.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000046.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000049.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000054.jpg',
'Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000056.jpg',
]



for img_path in img_paths:
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    predicted_class_index = np.argmax(result)

    class_labels = {v: k for k, v in training_set.class_indices.items()}
    class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus',
                   'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

    prediction = class_names[predicted_class_index]
    print(f"Resultado predicción: {prediction}")
