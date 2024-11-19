import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import standard_model, ResNet_model

#tf.__version__

#Prepocessing trainingset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../data/external/Skin cancer ISIC The International Skin Imaging Collaboration/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

#Prepocessing testset
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('../data/external/Skin cancer ISIC The International Skin Imaging Collaboration/Test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')




# Floder rute
test_dir = '../data/external/Skin cancer ISIC The International Skin Imaging Collaboration/Test'

# Load pretrained model
#pre_trained_model = tf.keras.models.load_model('standard_model_skin_cancer.h5')

#Load model
cnn = standard_model.standard_model(training_set, test_set)

# Obtener los nombres de las clases del conjunto de entrenamiento
class_labels = {v: k for k, v in training_set.class_indices.items()}
class_names = list(class_labels.values())
right_predictions = 0
total_predicctions = 0
# Recorrer cada subcarpeta y cada archivo de imagen
for class_folder in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_folder)
    if os.path.isdir(class_path):
        print(f"\nPredictions for the class: {class_folder}")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                # Loading image
                test_image = image.load_img(img_path, target_size=(64, 64))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0) / 255.0

                # Realizar la predicción
                result = pre_trained_model.predict(test_image)
                predicted_class_index = np.argmax(result)
                prediction = class_names[predicted_class_index]

                print(f"Imagen: {img_name} - Prediction: {prediction}")
                total_predicctions += 1
                if prediction == class_folder:
                    right_predictions += 1



print("Accuracy: ", right_predictions/total_predicctions)

