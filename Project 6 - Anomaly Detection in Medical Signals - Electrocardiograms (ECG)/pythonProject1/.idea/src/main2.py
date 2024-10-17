import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Directorio que contiene los archivos CSV
data_dir = '/Users/andreuolaortua/code/Machine learning/00 - Projects/Project 6 - Anomaly Detection in Medical Signals - Electrocardiograms (ECG)/pythonProject1/.idea/src/MIT-BIH Arrhythmia database/'

# Definir las derivaciones de interés
desired_columns = ['MLII', 'V1', 'V2', 'V3', 'V4', 'V5']

# Cargar todos los archivos CSV
def load_data_from_directory(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            data = pd.read_csv(filepath)
            
            # Verificamos qué derivaciones están disponibles
            available_columns = data.columns.intersection(desired_columns)
            
            # Creamos una matriz con las derivaciones disponibles (rellenando con ceros si faltan)
            signals = np.zeros((len(data), len(desired_columns)))  # Inicializamos con ceros
            for i, col in enumerate(desired_columns):
                if col in available_columns:
                    signals[:, i] = data[col].values  # Copiamos los valores si la columna está disponible
            
            all_data.append(signals)  # Agregamos las señales a la lista
    
    return np.vstack(all_data)  # Unimos todas las señales

# Cargar los datos
ecg_data = load_data_from_directory(data_dir)

# Parámetros
sequence_length = 200  # Longitud de las ventanas para cada señal
step = 50  # Salto entre ventanas
n_samples = len(ecg_data)
n_features = ecg_data.shape[1]  # Número de derivaciones/columnas

# Crear las ventanas deslizantes de datos
X = []
for i in range(0, n_samples - sequence_length, step):
    X.append(ecg_data[i:i + sequence_length])
X = np.array(X)

# Etiquetas simuladas (0 para normal, 1 para anormal)
y = np.random.randint(0, 2, len(X))

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las señales
scaler = StandardScaler()
X_train = X_train.reshape(-1, X_train.shape[-1])  # Aplanar para normalizar
X_train = scaler.fit_transform(X_train)
X_train = X_train.reshape(-1, sequence_length, n_features)  # Restaurar la forma original con las derivaciones correctas

X_test = X_test.reshape(-1, X_test.shape[-1])
X_test = scaler.transform(X_test)
X_test = X_test.reshape(-1, sequence_length, n_features)

# Construcción del modelo CNN
model = Sequential()

# Primera capa convolucional 1D
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# Segunda capa convolucional
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# Aplanar los datos
model.add(Flatten())

# Capa densa completamente conectada
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida (2 clases: normal o anormal)
model.add(Dense(1, activation='sigmoid'))

# Compilación del modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluación del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión en los datos de prueba: {accuracy * 100:.2f}%')

# Graficar el entrenamiento
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
