import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np

# Load the .h5 model
model = tf.keras.models.load_model(
    'C:/Users/Gelo/Documents/Programming/plant-disease-detection/public/models/Potato_Model.h5')

target_dir = 'C:/Users/Gelo/Documents/Programming/plant-disease-detection/public/converted_model'

# Convert the model to JSON
# model_json = model.to_json()

# Convert the Keras model to TensorFlow.js format
tfjs_version = '3.5.0'  # Specify an older version of TensorFlow.js
# Convert the Keras model to TensorFlow.js format (.json)
tfjs.converters.save_keras_model(model, target_dir, save_format='tfjs')


""" # Save the JSON model to a file
with open('Potato_Model.json', 'w') as json_file:
    json_file.write(model_json)

# Save the model weights to binary format
model.save_weights('Potato_Model_weights.bin') """
