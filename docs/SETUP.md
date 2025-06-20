# Plant Disease Classification - Setup Guide

## Quick Start

1. Clone the repository
2. Create virtual environment: `python -m venv plant-disease-env`
3. Activate environment: `source plant-disease-env/bin/activate` (Linux/Mac) or `plant-disease-env\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Setup dataset: `python src/preprocessing/dataset_setup.py`
6. Train model: `python src/models/train_model.py`
7. Run app: `python app.py`

## Detailed Instructions

See the main README.md for complete setup instructions.

## Troubleshooting

- If TensorFlow installation fails, try: `pip install tensorflow-cpu`
- For GPU support, install: `pip install tensorflow-gpu`
- If model training is slow, reduce BATCH_SIZE in config.py
EOF

cat > docs/API.md << 'EOF'
# API Documentation

## Endpoints

### POST /api/predict
Upload an image for plant disease prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "success": true,
  "predicted_class": "Tomato___Early_blight",
  "confidence": 0.92,
  "class_name": "Tomato - Early blight"
}