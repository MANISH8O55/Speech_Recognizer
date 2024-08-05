import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# Example model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Path to save the model
save_path = 'D:/Manish/Nuero/SER/Speech-Emotion-Recognizer/model/1'

# Save the model in the TensorFlow SavedModel format
tf.saved_model.save(model, save_path)


