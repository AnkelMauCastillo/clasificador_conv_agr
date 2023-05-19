import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Crear el modelo
model = Sequential()

# Agregar una capa de convolución con 32 filtros, tamaño de kernel 3x3 y función de activación ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Agregar una capa de agrupamiento (max pooling) con tamaño de ventana 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Agregar otra capa de convolución con 64 filtros y función de activación ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))

# Agregar otra capa de agrupamiento (max pooling) con tamaño de ventana 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar los mapas de características
model.add(Flatten())

# Agregar una capa completamente conectada con 128 unidades y función de activación ReLU
model.add(Dense(128, activation='relu'))

# Agregar una capa de salida completamente conectada con 10 unidades (correspondientes a las 10 clases de dígitos)
# y función de activación softmax para obtener probabilidades de clasificación
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluar el modelo en el conjunto de prueba
model.evaluate(x_test, y_test)

# Exportar el modelo
model.save('modelo_mnist.h5')
