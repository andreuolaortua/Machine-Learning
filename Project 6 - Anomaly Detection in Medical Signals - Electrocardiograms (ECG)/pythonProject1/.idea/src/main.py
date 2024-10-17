import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Cargar el dataset (suponiendo que es un archivo CSV con las señales)
# df = pd.read_csv('ruta_a_tu_ecg_dataset.csv')

# Para este ejemplo, voy a crear un dataset de prueba (simulado)
# Suponiendo que cada fila en 'X' es una señal de ECG y 'y' es la etiqueta (0: normal, 1: anormal)
X = np.random.randn(1000, 500)  # 1000 muestras, cada una con 500 puntos de datos de la señal
y = np.random.randint(0, 2, 1000)  # Etiquetas binarias (0 o 1)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las señales de ECG
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cambiar la forma de los datos para que sean compatibles con Conv1D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Crear el modelo CNN
model = Sequential()

# Capa convolucional 1D
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# Segunda capa convolucional
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# Aplanar la salida para conectarla a las capas densas
model.add(Flatten())

# Capa densa completamente conectada
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida (2 clases: normal o anormal)
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión en los datos de prueba: {accuracy * 100:.2f}%')

# Graficar la precisión y la pérdida del modelo
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()
