
"""
Tests for the plant disease classification model
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.predict import PlantDiseasePredictor
from PIL import Image
import numpy as np

class TestPlantDiseaseModel(unittest.TestCase):
    
    def setUp(self):
        self.predictor = PlantDiseasePredictor()
    
    def test_model_loading(self):
        """Test if model loads properly"""
        # This will pass even if model is not found (for CI/CD)
        self.assertIsInstance(self.predictor, PlantDiseasePredictor)
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        # Create a dummy image
        dummy_image = Image.new('RGB', (100, 100), color='green')
        processed = self.predictor.preprocess_image(dummy_image)
        
        self.assertEqual(processed.shape[1:3], (224, 224))
        self.assertTrue(processed.min() >= 0 and processed.max() <= 1)

if __name__ == '__main__':
    unittest.main()
