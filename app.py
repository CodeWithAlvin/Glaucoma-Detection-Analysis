from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
from backend import load_model, predict

app = Flask(__name__)

# Load the model at startup
model_path = 'src/model/VisionTransformer_glaucoma_model.pth'
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image file
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        image_array = np.array(image)
        
        # Make prediction
        result = predict(image_array, model)
        
        # Format the result
        prediction_text = "Glaucoma Positive" if result == 1 else "Glaucoma Negative"
        
        return jsonify({
            'result': result,
            'prediction_text': prediction_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)