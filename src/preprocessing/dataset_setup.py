"""
Dataset download and setup utilities
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import shutil
from config import Config

def setup_dataset():
    """Setup the PlantVillage dataset"""
    print("🌱 Setting up Plant Disease Dataset...")
    
    # Create directories
    Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)
    
    print("""
    📋 Dataset Setup Instructions:
    
    1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    2. Download the dataset (requires Kaggle account)
    3. Extract to: data/raw/dataset/
    
    Or run: kaggle datasets download -d abdallahalidev/plantvillage-dataset
    """)

if __name__ == "__main__":
    setup_dataset()
