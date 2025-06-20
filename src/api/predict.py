"""
Plant Disease Prediction API
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
from flask import jsonify
from config import Config

class PlantDiseasePredictor:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if Config.MODEL_PATH.exists():
                self.model = tf.keras.models.load_model(Config.MODEL_PATH)
                
                with open(Config.MODEL_INFO_PATH, 'r') as f:
                    model_info = json.load(f)
                    self.class_names = model_info['class_names']
                
                self.model_loaded = True
                print("✅ Model loaded successfully")
            else:
                print("⚠️ Model not found. Please train the model first.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(Config.IMAGE_SIZE)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_api(self, request):
        """API endpoint for predictions"""
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))
            
            if not self.model_loaded:
                return jsonify({'error': 'Model not loaded'}), 500
            
            # Preprocess and predict
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image)
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_name': predicted_class.replace('___', ' - ').replace('_', ' ')
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
