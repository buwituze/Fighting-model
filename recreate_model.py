# recreate_model.py
import tensorflow as tf
import numpy as np
import os
import h5py
from datetime import datetime

# Configuration
IMG_SIZE = 64
FRAMES_PER_VIDEO = 16
MODEL_PATH = "fight_detection_model_optimized.h5" 
NEW_MODEL_PATH = "fight_detection_model_v2_compatible.h5" 
BACKUP_MODEL_PATH = f"fight_detection_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

def create_compatible_model():
    """Create a compatible 3D CNN model for fight detection"""
    
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)),
        
        # First 3D Convolutional Block
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second 3D Convolutional Block
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third 3D Convolutional Block
        tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.GlobalAveragePooling3D(),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_simple_model():
    """Create a simpler model that matches the original architecture more closely"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def backup_existing_model():
    if os.path.exists(NEW_MODEL_PATH):
        try:
            import shutil
            shutil.copy2(NEW_MODEL_PATH, BACKUP_MODEL_PATH)
            print(f"✓ Existing model backed up as: {BACKUP_MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Could not backup existing model: {e}")
            return False
    return True

def extract_weights_from_h5():
    try:
        print("Attempting to extract weights directly from H5 file...")
        weights_dict = {}
        
        with h5py.File(MODEL_PATH, 'r') as f:
            print("H5 file structure:")
            def print_structure(name):
                print(f"  {name}")
            f.visit(print_structure)
            
            if 'model_weights' in f:
                model_weights = f['model_weights']
                for layer_name in model_weights:
                    layer_weights = []
                    layer_group = model_weights[layer_name]
                    for weight_name in layer_group:
                        weight_data = layer_group[weight_name][:]
                        layer_weights.append(weight_data)
                    if layer_weights:
                        weights_dict[layer_name] = layer_weights
                        print(f"Extracted weights for layer: {layer_name}")
        
        return weights_dict
        
    except Exception as e:
        print(f"Could not extract weights from H5 file: {e}")
        return {}

def test_original_model():

    try:
        original_model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Original model loaded successfully")
        print(f"  Input shape: {original_model.input_shape}")
        print(f"  Output shape: {original_model.output_shape}")
        
        dummy_input = np.random.random((1, FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3))
        prediction = original_model.predict(dummy_input, verbose=0)
        print(f"  Test prediction: {prediction[0][0]:.4f}")
        
        return original_model
        
    except Exception as e:
        print(f"❌ Could not load original model: {e}")
        return None

def main():
    print("Fight Detection Model Recreation ")
    print(f"New model path: {NEW_MODEL_PATH}")
    print(f"Backup path: {BACKUP_MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Original model file not found: {MODEL_PATH}")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                print(f"  - {file}")
        return
    
    print(f"✓ Original model file found ({os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB)")
    
    if not backup_existing_model():
        response = input("Continue without backup? (y/n): ")
        if response.lower() != 'y':
            return
    
    original_model = test_original_model()
    
    print("\nCreating compatible model...")
    
    try:
        print("Attempting simple model architecture...")
        model = create_simple_model()
        print("✓ Simple model created successfully")
        model_type = "simple"
    except Exception as e:
        print(f"Simple model failed: {e}")
        try:
            print("Attempting complex model architecture...")
            model = create_compatible_model()
            print("✓ Complex model created successfully")
            model_type = "complex"
        except Exception as e2:
            print(f"❌ Both model creation attempts failed: {e2}")
            return
    
    print(f"\nModel Summary ({model_type}):")
    model.summary()
    
    print(f"\nSaving model to {NEW_MODEL_PATH}...")
    try:
        model.save(NEW_MODEL_PATH, save_format='h5')
        print("✓ Model saved successfully")
        print(f"  File size: {os.path.getsize(NEW_MODEL_PATH) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print("\nTesting model loading...")
    try:
        test_model = tf.keras.models.load_model(NEW_MODEL_PATH)
        print("✓ Model loaded successfully!")
        
        print(f"  Input shape: {test_model.input_shape}")
        print(f"  Output shape: {test_model.output_shape}")
        
        print("\nTesting prediction")
        dummy_input = np.random.random((1, FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3))
        prediction = test_model.predict(dummy_input, verbose=0)
        print(f"✓ Prediction test successful: {prediction[0][0]:.4f}")

        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    main()