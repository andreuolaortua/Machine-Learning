# Skin Cancer Classification Model

This project aims to classify images of skin lesions into multiple categories using a Convolutional Neural Network (CNN). The model is trained on the **Skin Cancer ISIC dataset** and is capable of distinguishing between 9 types of skin conditions.

---

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Performance](#performance)
- [Future Improvements](#future-improvements)

---

## Dataset

The dataset used is the **Skin Cancer ISIC: The International Skin Imaging Collaboration**. It consists of categorized images of various skin conditions:

### Classes:
1. Melanoma
2. Nevus
3. Basal Cell Carcinoma
4. Pigmented Benign Keratosis
5. Actinic Keratosis
6. Squamous Cell Carcinoma
7. Vascular Lesion
8. Seborrheic Keratosis
9. Dermatofibroma

- **Train/Test Split**: The dataset is divided into training and testing sets located in separate directories.

---

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using TensorFlow/Keras. The architecture consists of:

1. **Input Layer**:
   - Input shape: `(64, 64, 3)` (RGB images resized to 64x64 pixels).

2. **Convolutional Layers**:
   - Four convolutional layers with ReLU activation and 32 filters each.
   - Max-pooling applied after each convolution to reduce spatial dimensions.

3. **Fully Connected Layers**:
   - Flattened layer followed by a dense layer with 128 units and ReLU activation.
   - Output layer with 9 units and softmax activation for multi-class classification.

4. **Compilation**:
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Metric: Accuracy

---

## Preprocessing

1. **Data Augmentation**:
   - Rescaling: Images are normalized by dividing pixel values by 255.
   - Augmentation techniques: Shear, zoom, and horizontal flips to prevent overfitting.

2. **Input Dimensions**:
   - All images are resized to `(64, 64)`.

---

## Training

- **Epochs**: 25
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

The model is trained on the augmented training set and validated using the test set.

---

## Evaluation

After training, the model is evaluated on the test dataset by predicting the class of each image. The final accuracy is calculated as the ratio of correctly classified images to the total number of images.
