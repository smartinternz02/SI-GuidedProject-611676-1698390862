from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("k1.h5")

# Define the class labels
class_labels = ['cataract', 'diabetic retinopathy', 'glaucoma', 'normal']

# Create an 'uploads' directory if it doesn't exist
uploads_dir = os.path.join(app.root_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        img_file = request.files['file']

        # Save the uploaded image with a unique name
        img_path = os.path.join(uploads_dir, 'uploaded_image.jpg')
        img_file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Make predictions
        preds = model.predict(x)

        # Get the index of the predicted class
        pred = np.argmax(preds, axis=1)

        # Get the predicted class label
        result = class_labels[pred[0]]

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
