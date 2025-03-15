from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(_name_)
model = load_model("model/rice_model.h5")

# Define rice categories
categories = ['Basmati', 'Jasmine', 'Arborio', 'Others']  # Adjust based on your dataset

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = categories[np.argmax(prediction)]
    
    return jsonify({"rice_type": predicted_class})

if _name_ == '_main_':
    app.run(debug=True)
