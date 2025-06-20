"""
Plant Disease Classification Model Training
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from config import Config

class PlantDiseaseTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = []
        
    def create_data_generators(self):
        """Create data generators for training"""
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=Config.VALIDATION_SPLIT
        )
        
        train_generator = train_datagen.flow_from_directory(
            Config.DATASET_PATH,
            target_size=Config.IMAGE_SIZE,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            Config.DATASET_PATH,
            target_size=Config.IMAGE_SIZE,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        return train_generator, val_generator
    
    def create_model(self, num_classes):
        """Create CNN model"""
        model = keras.Sequential([
            keras.layers.Input(shape=(*Config.IMAGE_SIZE, 3)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        """Train the model"""
        print("🚀 Starting model training...")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Create model
        num_classes = len(self.class_names)
        self.model = self.create_model(num_classes)
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.EPOCHS,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10),
                keras.callbacks.ReduceLROnPlateau(patience=5)
            ]
        )
        
        # Save model
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(Config.MODEL_PATH)
        
        # Save model info
        model_info = {
            'class_names': self.class_names,
            'num_classes': num_classes,
            'image_size': Config.IMAGE_SIZE,
            'accuracy': max(self.history.history['val_accuracy'])
        }
        
        with open(Config.MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model saved to {Config.MODEL_PATH}")
        
if __name__ == "__main__":
    trainer = PlantDiseaseTrainer()
    trainer.train()
