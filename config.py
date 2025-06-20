"""
Configuration settings for the Plant Disease Classification app
"""

import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'models'
    STATIC_DIR = BASE_DIR / 'static'
    
    # Model settings
    MODEL_PATH = MODEL_DIR / 'plant_disease_model.h5'
    MODEL_INFO_PATH = MODEL_DIR / 'model_info.json'
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # Training settings
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Dataset settings
    DATASET_PATH = DATA_DIR / 'raw' / 'dataset'
    PROCESSED_DATA_PATH = DATA_DIR / 'processed'
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
