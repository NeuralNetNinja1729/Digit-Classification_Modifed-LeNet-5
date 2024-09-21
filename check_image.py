import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("./models/digit_classification_cnn_with_BatchNormalizationAndDropout_2.h5")

path = "./testing_images/image_8.png" 
image = Image.open(path).convert("L") 

 
image = image.resize((32, 32))

image_array = np.array(image).reshape((1, 32, 32, 1))

predictions = model.predict(image_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence_score = np.max(predictions, axis=1)[0]

print(f"Predicted Digit: {predicted_class}.")
print(f'Confidence Score: {round(confidence_score, 2)}')
