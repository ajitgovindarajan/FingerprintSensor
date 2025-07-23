import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt

class FingerprintAIProcessor:
    """
    Python-based AI components for fingerprint recognition system.
    Handles deep learning, feature extraction, and model training.
    """
    
    def __init__(self, model_path: str = None):
        self.feature_extractor = None
        self.similarity_model = None
        self.quality_model = None
        self.liveness_model = None
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_models(model_path)
        else:
            self._build_models()
    
    def _build_models(self):
        """Build neural network models for different components."""
        # Feature extraction CNN
        self.feature_extractor = self._build_feature_extractor()
        
        # Similarity comparison model
        self.similarity_model = self._build_similarity_model()
        
        # Image quality assessment model
        self.quality_model = self._build_quality_model()
        
        # Liveness detection model
        self.liveness_model = self._build_liveness_model()
    
    def _build_feature_extractor(self) -> keras.Model:
        """Build CNN for deep feature extraction from fingerprint images."""
        model = keras.Sequential([
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Global average pooling and dense layers
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, name='feature_vector')  # Feature vector output
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _build_similarity_model(self) -> keras.Model:
        """Build model to compare feature vectors and output similarity score."""
        input_a = keras.layers.Input(shape=(128,), name='features_a')
        input_b = keras.layers.Input(shape=(128,), name='features_b')
        
        # Compute absolute difference and concatenate
        diff = keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([input_a, input_b])
        concat = keras.layers.Concatenate()([input_a, input_b, diff])
        
        # Dense layers for similarity computation
        x = keras.layers.Dense(256, activation='relu')(concat)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        
        # Output similarity score
        similarity = keras.layers.Dense(1, activation='sigmoid', name='similarity')(x)
        
        model = keras.Model(inputs=[input_a, input_b], outputs=similarity)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _build_quality_model(self) -> keras.Model:
        """Build model to assess fingerprint image quality."""
        model = keras.Sequential([
            keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(256, 256, 1)),
            keras.layers.MaxPooling2D((4, 4)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid', name='quality_score')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_liveness_model(self) -> keras.Model:
        """Build model for liveness detection (anti-spoofing)."""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid', name='liveness_score')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess fingerprint image for AI models."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        image = cv2.resize(image, (256, 256))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply((image * 255).astype(np.uint8)) / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features from fingerprint image using CNN."""
        processed_image = self.preprocess_image(image)
        features = self.feature_extractor.predict(processed_image, verbose=0)
        return features[0]  # Remove batch dimension
    
    def assess_image_quality(self, image: np.ndarray) -> float:
        """Assess the quality of a fingerprint image."""
        processed_image = self.preprocess_image(image)
        quality_score = self.quality_model.predict(processed_image, verbose=0)
        return float(quality_score[0][0])
    
    def detect_liveness(self, image: np.ndarray) -> Tuple[bool, float]:
        """Detect if the fingerprint is from a live finger (anti-spoofing)."""
        processed_image = self.preprocess_image(image)
        liveness_score = self.liveness_model.predict(processed_image, verbose=0)
        score = float(liveness_score[0][0])
        is_live = score > 0.5
        return is_live, score
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors."""
        # Reshape for model input
        feat1 = features1.reshape(1, -1)
        feat2 = features2.reshape(1, -1)
        
        # Use neural network similarity model
        nn_similarity = self.similarity_model.predict([feat1, feat2], verbose=0)
        
        # Also calculate cosine similarity as backup
        cosine_sim = cosine_similarity(feat1, feat2)[0][0]
        
        # Combine both similarities
        combined_similarity = 0.7 * float(nn_similarity[0][0]) + 0.3 * cosine_sim
        
        return combined_similarity
    
    def train_feature_extractor(self, images: List[np.ndarray], labels: List[int], 
                              validation_split: float = 0.2, epochs: int = 50):
        """Train the feature extraction model."""
        # Preprocess training data
        X = []
        for img in images:
            processed = self.preprocess_image(img)
            X.append(processed[0])  # Remove batch dimension
        
        X = np.array(X)
        y = np.array(labels)
        
        # Create training pairs for similarity learning
        X_pairs, y_pairs = self._create_training_pairs(X, y)
        
        # Train the feature extractor with similarity learning
        history = self.similarity_model.fit(
            X_pairs, y_pairs,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def _create_training_pairs(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Create pairs for similarity learning."""
        pairs = []
        labels = []
        
        n_samples = len(X)
        
        # Create positive pairs (same finger)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if y[i] == y[j]:  # Same finger
                    feat1 = self.feature_extractor.predict(np.expand_dims(X[i], 0), verbose=0)[0]
                    feat2 = self.feature_extractor.predict(np.expand_dims(X[j], 0), verbose=0)[0]
                    pairs.append([feat1, feat2])
                    labels.append(1)
        
        # Create negative pairs (different fingers)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if y[i] != y[j]:  # Different fingers
                    feat1 = self.feature_extractor.predict(np.expand_dims(X[i], 0), verbose=0)[0]
                    feat2 = self.feature_extractor.predict(np.expand_dims(X[j], 0), verbose=0)[0]
                    pairs.append([feat1, feat2])
                    labels.append(0)
                    
                    # Balance the dataset
                    if len([l for l in labels if l == 0]) >= len([l for l in labels if l == 1]):
                        break
        
        # Convert to proper format
        pairs = np.array(pairs)
        X_a = pairs[:, 0]
        X_b = pairs[:, 1]
        y_pairs = np.array(labels)
        
        return [X_a, X_b], y_pairs
    
    def save_models(self, path: str):
        """Save all trained models."""
        os.makedirs(path, exist_ok=True)
        
        self.feature_extractor.save(os.path.join(path, 'feature_extractor.h5'))
        self.similarity_model.save(os.path.join(path, 'similarity_model.h5'))
        self.quality_model.save(os.path.join(path, 'quality_model.h5'))
        self.liveness_model.save(os.path.join(path, 'liveness_model.h5'))
        
        # Save scaler
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_models(self, path: str):
        """Load pre-trained models."""
        self.feature_extractor = keras.models.load_model(os.path.join(path, 'feature_extractor.h5'))
        self.similarity_model = keras.models.load_model(os.path.join(path, 'similarity_model.h5'))
        self.quality_model = keras.models.load_model(os.path.join(path, 'quality_model.h5'))
        self.liveness_model = keras.models.load_model(os.path.join(path, 'liveness_model.h5'))
        
        # Load scaler
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)


class FingerprintDatabase:
    """Simple database for storing fingerprint templates."""
    
    def __init__(self, db_path: str = "fingerprint_db.pkl"):
        self.db_path = db_path
        self.templates = {}
        self.load_database()
    
    def store_template(self, user_id: str, features: np.ndarray, metadata: Dict = None):
        """Store fingerprint template for a user."""
        self.templates[user_id] = {
            'features': features,
            'metadata': metadata or {},
            'created_at': np.datetime64('now')
        }
        self.save_database()
    
    def get_template(self, user_id: str) -> Optional[Dict]:
        """Retrieve fingerprint template for a user."""
        return self.templates.get(user_id)
    
    def get_all_templates(self) -> Dict:
        """Get all stored templates."""
        return self.templates
    
    def delete_template(self, user_id: str) -> bool:
        """Delete a user's fingerprint template."""
        if user_id in self.templates:
            del self.templates[user_id]
            self.save_database()
            return True
        return False
    
    def save_database(self):
        """Save database to file."""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.templates, f)
    
    def load_database(self):
        """Load database from file."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.templates = pickle.load(f)


class FingerprintRecognitionSystem:
    """Main system combining all components."""
    
    def __init__(self, model_path: str = None, db_path: str = "fingerprint_db.pkl"):
        self.ai_processor = FingerprintAIProcessor(model_path)
        self.database = FingerprintDatabase(db_path)
        self.quality_threshold = 0.6
        self.similarity_threshold = 0.8
    
    def enroll_user(self, user_id: str, fingerprint_images: List[np.ndarray]) -> Dict:
        """Enroll a new user with multiple fingerprint samples."""
        if not fingerprint_images:
            return {"success": False, "error": "No images provided"}
        
        quality_scores = []
        feature_vectors = []
        
        for img in fingerprint_images:
            # Check image quality
            quality = self.ai_processor.assess_image_quality(img)
            quality_scores.append(quality)
            
            if quality < self.quality_threshold:
                continue
            
            # Check liveness
            is_live, liveness_score = self.ai_processor.detect_liveness(img)
            if not is_live:
                continue
            
            # Extract features
            features = self.ai_processor.extract_deep_features(img)
            feature_vectors.append(features)
        
        if not feature_vectors:
            return {"success": False, "error": "No good quality live images found"}
        
        # Create master template from multiple samples
        master_template = np.mean(feature_vectors, axis=0)
        
        # Store in database
        metadata = {
            "num_samples": len(feature_vectors),
            "avg_quality": np.mean(quality_scores),
            "enrollment_date": str(np.datetime64('now'))
        }
        
        self.database.store_template(user_id, master_template, metadata)
        
        return {
            "success": True,
            "user_id": user_id,
            "samples_used": len(feature_vectors),
            "average_quality": np.mean(quality_scores)
        }
    
    def authenticate_user(self, fingerprint_image: np.ndarray) -> Dict:
        """Authenticate a user against enrolled templates."""
        # Check image quality
        quality = self.ai_processor.assess_image_quality(fingerprint_image)
        if quality < self.quality_threshold:
            return {
                "authenticated": False,
                "error": "Poor image quality",
                "quality_score": quality
            }
        
        # Check liveness
        is_live, liveness_score = self.ai_processor.detect_liveness(fingerprint_image)
        if not is_live:
            return {
                "authenticated": False,
                "error": "Liveness detection failed",
                "liveness_score": liveness_score
            }
        
        # Extract features
        query_features = self.ai_processor.extract_deep_features(fingerprint_image)
        
        # Compare against all enrolled templates
        best_match = None
        best_similarity = 0
        
        for user_id, template_data in self.database.get_all_templates().items():
            stored_features = template_data['features']
            similarity = self.ai_processor.calculate_similarity(query_features, stored_features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user_id
        
        # Check if best match exceeds threshold
        if best_similarity >= self.similarity_threshold:
            return {
                "authenticated": True,
                "user_id": best_match,
                "confidence": best_similarity,
                "quality_score": quality,
                "liveness_score": liveness_score
            }
        else:
            return {
                "authenticated": False,
                "error": "No matching fingerprint found",
                "best_similarity": best_similarity,
                "quality_score": quality,
                "liveness_score": liveness_score
            }


# Example usage and testing
def main():
    """Example usage of the fingerprint recognition system."""
    # Initialize the system
    system = FingerprintRecognitionSystem()
    
    # Example: Load and process a fingerprint image
    # (In real use, this would come from a camera or sensor)
    # img = cv2.imread('fingerprint_sample.png', cv2.IMREAD_GRAYSCALE)
    
    print("Fingerprint Recognition System initialized")
    print("Available methods:")
    print("- system.enroll_user(user_id, [images])")
    print("- system.authenticate_user(image)")
    print("- system.ai_processor.assess_image_quality(image)")
    print("- system.ai_processor.detect_liveness(image)")


if __name__ == "__main__":
    main()