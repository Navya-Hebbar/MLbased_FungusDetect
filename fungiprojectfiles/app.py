from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\navya\Downloads\fungi_detector_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise ValueError("Model could not be loaded")

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        # Load and preprocess the image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))  # Resize to model's input size
        img_array = np.array(img)      # Convert to NumPy array
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Add batch dimension and normalize

        # Perform the prediction
        prediction = model.predict(img_array)

        # Determine result
        if prediction[0] > 0.5:
            result = "No Fungus Detected"
        else:
            result = "Fungus Detected"

        # Send prediction response
        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
