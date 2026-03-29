import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

# Load the trained model
def load_model(folder):
    W1 = np.load(f"{folder}/W1.npy")
    b1 = np.load(f"{folder}/b1.npy")
    W2 = np.load(f"{folder}/W2.npy")
    b2 = np.load(f"{folder}/b2.npy")
    return W1, b1, W2, b2

# Predict function
def predict_pothole(image, model_folder):
    W1, b1, W2, b2 = load_model(model_folder)
    image = image.resize((64, 64)).convert("L")
    image_array = np.array(image).reshape(1, -1) / 255.0  # Normalize and flatten
    
    def forward_propagation(X, W1, b1, W2, b2):
        def relu(Z): return np.maximum(0, Z)
        def sigmoid(Z): return 1 / (1 + np.exp(-Z))
        Z1 = np.dot(W1, X.T) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        return A2

    A2 = forward_propagation(image_array, W1, b1, W2, b2)
    prediction = (A2 > 0.5).astype(int).item()
    return "Pothole Detected." if prediction == 1 else "Normal Road."

# Flask app
app = Flask(__name__)
MODEL_FOLDER = "pothole_detection_model"  # Folder containing model weights

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded."
    
    file = request.files["file"]
    if file.filename == "":
        return "No file selected."
    
    image = Image.open(file)
    result = predict_pothole(image, MODEL_FOLDER)
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
