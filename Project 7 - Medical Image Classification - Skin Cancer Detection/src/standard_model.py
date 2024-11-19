import tensorflow as tf


def standard_model(training_set, test_set):

    #Initialising the CNN
    cnn = tf.keras.models.Sequential()
    #Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    #Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    #padding a second convolutional layer
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

    cnn.summary()

    cnn.save('standard_model_skin_cancer.h5')

    return cnn
