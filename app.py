
"""
Plant Disease Classification Web Application
Main Flask application file
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from src.api.predict import PlantDiseasePredictor

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = PlantDiseasePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    return predictor.predict_api(request)

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': predictor.model_loaded})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
