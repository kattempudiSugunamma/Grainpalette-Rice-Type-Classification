import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("model/rice_model.h5")

# Define rice categories
categories = ['Basmati', 'Jasmine', 'Arborio', 'Others']  # Adjust based on your dataset

def predict_rice(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]

    return predicted_class

# Test with an image
test_image = "test_rice.jpg"
predicted_rice = predict_rice(test_image)
print(f"Predicted Rice Type: {predicted_rice}")
