"""
This version utlized deep neural networks to better handle the data which is key for the fingerpritting that is critical
for security

First we need the neural network based handling sysem which will be handled through a class object

"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
import hashlib
import sqlite3
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime

class NeuralDataHandler:
    """Neural network-based data handling system for fingerprint templates."""
    
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
        self.autoencoder = None
        self.neural_hasher = None
        self.quality_predictor = None
        self.template_updater = None
        self.clustering_model = KMeans(n_clusters=50, random_state=42)
        self._build_neural_components()
    
    def _build_neural_components(self):
        """Build all neural network components for data handling."""
        self.autoencoder = self._build_autoencoder()
        self.neural_hasher = self._build_neural_hasher()
        self.quality_predictor = self._build_quality_predictor()
        self.template_updater = self._build_template_updater()
    
    def _build_autoencoder(self) -> keras.Model:
        """Build autoencoder for template compression."""
        # Encoder
        input_layer = keras.layers.Input(shape=(self.feature_dim,))
        encoded = keras.layers.Dense(64, activation='relu')(input_layer)
        encoded = keras.layers.BatchNormalization()(encoded)
        encoded = keras.layers.Dense(32, activation='relu')(encoded)
        encoded = keras.layers.BatchNormalization()(encoded)
        encoded = keras.layers.Dense(16, activation='relu', name='compressed_features')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(32, activation='relu')(encoded)
        decoded = keras.layers.BatchNormalization()(decoded)
        decoded = keras.layers.Dense(64, activation='relu')(decoded)
        decoded = keras.layers.BatchNormalization()(decoded)
        decoded = keras.layers.Dense(self.feature_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return {'autoencoder': autoencoder, 'encoder': encoder}
    
    def _build_neural_hasher(self) -> keras.Model:
        """Build neural network for creating similarity-aware hash indices."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.feature_dim,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='tanh'),  # Hash output
            keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))  # Normalize for cosine similarity
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _build_quality_predictor(self) -> keras.Model:
        """Build network to predict template quality over time."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.feature_dim + 5,)),  # features + metadata
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid', name='quality_score')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_template_updater(self) -> keras.Model:
        """Build network for adaptive template updates."""
        # Inputs: old template, new features, confidence scores
        old_template = keras.layers.Input(shape=(self.feature_dim,), name='old_template')
        new_features = keras.layers.Input(shape=(self.feature_dim,), name='new_features')
        confidence = keras.layers.Input(shape=(1,), name='confidence')
        update_count = keras.layers.Input(shape=(1,), name='update_count')
        
        # Feature processing
        old_processed = keras.layers.Dense(64, activation='relu')(old_template)
        new_processed = keras.layers.Dense(64, activation='relu')(new_features)
        
        # Attention mechanism for weighted combination
        attention_input = keras.layers.Concatenate()([old_processed, new_processed, confidence, update_count])
        attention_weights = keras.layers.Dense(64, activation='relu')(attention_input)
        attention_weights = keras.layers.Dense(2, activation='softmax')(attention_weights)
        
        # Weighted combination
        old_weight = keras.layers.Lambda(lambda x: x[:, 0:1])(attention_weights)
        new_weight = keras.layers.Lambda(lambda x: x[:, 1:2])(attention_weights)
        
        weighted_old = keras.layers.Multiply()([old_template, old_weight])
        weighted_new = keras.layers.Multiply()([new_features, new_weight])
        
        updated_template = keras.layers.Add()([weighted_old, weighted_new])
        updated_template = keras.layers.Dense(self.feature_dim, activation='linear')(updated_template)
        
        model = keras.Model(
            inputs=[old_template, new_features, confidence, update_count],
            outputs=updated_template
        )
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def compress_template(self, template: np.ndarray) -> np.ndarray:
        """Compress template using autoencoder."""
        template_reshaped = template.reshape(1, -1)
        compressed = self.autoencoder['encoder'].predict(template_reshaped, verbose=0)
        return compressed[0]
    
    def decompress_template(self, compressed_template: np.ndarray) -> np.ndarray:
        """Decompress template using autoencoder."""
        compressed_reshaped = compressed_template.reshape(1, -1)
        decompressed = self.autoencoder['autoencoder'].predict(compressed_reshaped, verbose=0)
        return decompressed[0]
    
    def generate_neural_hash(self, template: np.ndarray) -> np.ndarray:
        """Generate neural hash for fast similarity-based lookups."""
        template_reshaped = template.reshape(1, -1)
        hash_vector = self.neural_hasher.predict(template_reshaped, verbose=0)
        return hash_vector[0]
    
    def predict_template_quality(self, template: np.ndarray, metadata: Dict) -> float:
        """Predict template quality using neural network."""
        # Extract metadata features
        meta_features = [
            metadata.get('usage_count', 0),
            metadata.get('days_since_creation', 0),
            metadata.get('avg_confidence', 0.5),
            metadata.get('successful_auths', 0),
            metadata.get('failed_auths', 0)
        ]
        
        # Combine template and metadata
        input_features = np.concatenate([template, meta_features])
        input_reshaped = input_features.reshape(1, -1)
        
        quality = self.quality_predictor.predict(input_reshaped, verbose=0)
        return float(quality[0][0])
    
    def update_template_adaptively(self, old_template: np.ndarray, new_features: np.ndarray, 
                                 confidence: float, update_count: int) -> np.ndarray:
        """Update template using neural network for optimal adaptation."""
        old_reshaped = old_template.reshape(1, -1)
        new_reshaped = new_features.reshape(1, -1)
        conf_reshaped = np.array([[confidence]])
        count_reshaped = np.array([[update_count]])
        
        updated = self.template_updater.predict([
            old_reshaped, new_reshaped, conf_reshaped, count_reshaped
        ], verbose=0)
        
        return updated[0]


class IntelligentFingerprintDatabase:
    """Neural network-enhanced database with smart indexing and compression."""
    
    def __init__(self, db_path: str = "intelligent_fp_db.sqlite", feature_dim: int = 128):
        self.db_path = db_path
        self.feature_dim = feature_dim
        self.neural_handler = NeuralDataHandler(feature_dim)
        self.hash_index = {}  # Neural hash to user_id mapping
        self.cluster_index = defaultdict(list)  # Cluster to user_ids mapping
        self.quality_cache = {}  # Cached quality scores
        
        self._init_database()
        self._load_indices()
    
    def _init_database(self):
        """Initialize SQLite database with neural enhancement tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main templates table with compression
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                user_id TEXT PRIMARY KEY,
                compressed_features BLOB,
                neural_hash BLOB,
                cluster_id INTEGER,
                metadata TEXT,
                quality_score REAL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                successful_auths INTEGER DEFAULT 0,
                failed_auths INTEGER DEFAULT 0
            )
        ''')
        
        # Authentication history for adaptive learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TIMESTAMP,
                confidence REAL,
                success BOOLEAN,
                feature_drift REAL,
                FOREIGN KEY (user_id) REFERENCES templates (user_id)
            )
        ''')
        
        # Neural model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT,
                timestamp TIMESTAMP,
                accuracy REAL,
                precision_val REAL,
                recall_val REAL,
                f1_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_indices(self):
        """Load neural hash and cluster indices into memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, neural_hash, cluster_id FROM templates')
        results = cursor.fetchall()
        
        for user_id, hash_blob, cluster_id in results:
            if hash_blob:
                neural_hash = pickle.loads(hash_blob)
                hash_key = self._hash_to_key(neural_hash)
                self.hash_index[hash_key] = user_id
                
            if cluster_id is not None:
                self.cluster_index[cluster_id].append(user_id)
        
        conn.close()
    
    def _hash_to_key(self, neural_hash: np.ndarray) -> str:
        """Convert neural hash to string key."""
        return hashlib.md5(neural_hash.tobytes()).hexdigest()
    
    def store_template(self, user_id: str, features: np.ndarray, metadata: Dict = None):
        """Store template with neural compression and intelligent indexing."""
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        metadata['created_at'] = datetime.now().isoformat()
        
        # Compress features
        compressed_features = self.neural_handler.compress_template(features)
        
        # Generate neural hash for fast lookup
        neural_hash = self.neural_handler.generate_neural_hash(features)
        hash_key = self._hash_to_key(neural_hash)
        
        # Determine cluster for grouping similar templates
        cluster_id = self._assign_cluster(features)
        
        # Predict initial quality
        quality_score = self.neural_handler.predict_template_quality(features, metadata)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO templates 
            (user_id, compressed_features, neural_hash, cluster_id, metadata, 
             quality_score, created_at, updated_at, usage_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            pickle.dumps(compressed_features),
            pickle.dumps(neural_hash),
            cluster_id,
            json.dumps(metadata),
            quality_score,
            datetime.now(),
            datetime.now(),
            0
        ))
        
        conn.commit()
        conn.close()
        
        # Update indices
        self.hash_index[hash_key] = user_id
        self.cluster_index[cluster_id].append(user_id)
        self.quality_cache[user_id] = quality_score
    
    def _assign_cluster(self, features: np.ndarray) -> int:
        """Assign features to a cluster for intelligent grouping."""
        features_reshaped = features.reshape(1, -1)
        
        # Use existing clustering if available, otherwise fit
        if hasattr(self.neural_handler.clustering_model, 'cluster_centers_'):
            cluster_id = self.neural_handler.clustering_model.predict(features_reshaped)[0]
        else:
            # Initialize with this sample
            cluster_id = 0
        
        return int(cluster_id)
    
    def get_template(self, user_id: str) -> Optional[Dict]:
        """Retrieve and decompress template."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT compressed_features, metadata, quality_score, usage_count,
                   successful_auths, failed_auths, created_at, updated_at
            FROM templates WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            compressed_features, metadata_json, quality_score, usage_count, \
            successful_auths, failed_auths, created_at, updated_at = result
            
            # Decompress features
            compressed = pickle.loads(compressed_features)
            features = self.neural_handler.decompress_template(compressed)
            
            return {
                'features': features,
                'metadata': json.loads(metadata_json) if metadata_json else {},
                'quality_score': quality_score,
                'usage_count': usage_count,
                'successful_auths': successful_auths,
                'failed_auths': failed_auths,
                'created_at': created_at,
                'updated_at': updated_at
            }
        
        return None
    
    def intelligent_search(self, query_features: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Use neural hashing and clustering for fast candidate retrieval."""
        candidates = []
        
        # Generate neural hash for query
        query_hash = self.neural_handler.generate_neural_hash(query_features)
        query_cluster = self._assign_cluster(query_features)
        
        # Get candidates from same cluster first (fast lookup)
        cluster_candidates = self.cluster_index.get(query_cluster, [])
        
        # Add candidates from neural hash similarity
        query_hash_key = self._hash_to_key(query_hash)
        if query_hash_key in self.hash_index:
            cluster_candidates.append(self.hash_index[query_hash_key])
        
        # Calculate actual similarities for candidates
        similarities = []
        for user_id in set(cluster_candidates):  # Remove duplicates
            template_data = self.get_template(user_id)
            if template_data:
                stored_features = template_data['features']
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    stored_features.reshape(1, -1)
                )[0][0]
                similarities.append((user_id, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def adaptive_update_template(self, user_id: str, new_features: np.ndarray, 
                               confidence: float, success: bool):
        """Adaptively update template based on authentication results."""
        template_data = self.get_template(user_id)
        if not template_data:
            return False
        
        old_features = template_data['features']
        usage_count = template_data['usage_count']
        
        # Update template using neural network
        updated_features = self.neural_handler.update_template_adaptively(
            old_features, new_features, confidence, usage_count
        )
        
        # Update metadata
        metadata = template_data['metadata']
        metadata['last_update'] = datetime.now().isoformat()
        
        # Store updated template
        self.store_template(user_id, updated_features, metadata)
        
        # Update usage statistics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE templates 
            SET usage_count = usage_count + 1,
                successful_auths = successful_auths + ?,
                failed_auths = failed_auths + ?,
                updated_at = ?
            WHERE user_id = ?
        ''', (1 if success else 0, 0 if success else 1, datetime.now(), user_id))
        
        # Log authentication history
        feature_drift = np.linalg.norm(old_features - new_features)
        cursor.execute('''
            INSERT INTO auth_history (user_id, timestamp, confidence, success, feature_drift)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, datetime.now(), confidence, success, feature_drift))
        
        conn.commit()
        conn.close()
        
        return True
    
    def cleanup_low_quality_templates(self, quality_threshold: float = 0.3):
        """Remove templates with persistently low quality."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find low quality templates
        cursor.execute('''
            SELECT user_id FROM templates 
            WHERE quality_score < ? AND usage_count > 10
        ''', (quality_threshold,))
        
        low_quality_users = [row[0] for row in cursor.fetchall()]
        
        # Remove them
        for user_id in low_quality_users:
            self.delete_template(user_id)
            print(f"Removed low quality template for user: {user_id}")
        
        conn.close()
        return len(low_quality_users)
    
    def get_analytics(self) -> Dict:
        """Get database analytics and performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Template statistics
        cursor.execute('SELECT COUNT(*), AVG(quality_score), AVG(usage_count) FROM templates')
        total_templates, avg_quality, avg_usage = cursor.fetchone()
        
        # Authentication statistics
        cursor.execute('''
            SELECT COUNT(*), 
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                   AVG(confidence),
                   AVG(feature_drift)
            FROM auth_history
        ''')
        total_auths, successful_auths, avg_confidence, avg_drift = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_templates': total_templates,
            'average_quality': avg_quality,
            'average_usage': avg_usage,
            'total_authentications': total_auths,
            'success_rate': successful_auths / total_auths if total_auths > 0 else 0,
            'average_confidence': avg_confidence,
            'average_feature_drift': avg_drift,
            'compression_ratio': 16 / self.feature_dim  # Compressed to 16 features
        }
    
    def delete_template(self, user_id: str) -> bool:
        """Delete template and clean up indices."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM templates WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM auth_history WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        # Clean up indices
        for hash_key, stored_user_id in list(self.hash_index.items()):
            if stored_user_id == user_id:
                del self.hash_index[hash_key]
                break
        
        for cluster_id, user_list in self.cluster_index.items():
            if user_id in user_list:
                user_list.remove(user_id)
        
        if user_id in self.quality_cache:
            del self.quality_cache[user_id]
        
        return True


class EnhancedFingerprintAIProcessor:
    """Enhanced AI processor with neural data handling integration."""
    
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
        """Build enhanced neural network models."""
        # Feature extraction CNN with attention
        self.feature_extractor = self._build_attention_feature_extractor()
        self.similarity_model = self._build_enhanced_similarity_model()
        self.quality_model = self._build_enhanced_quality_model()
        self.liveness_model = self._build_enhanced_liveness_model()
    
    def _build_attention_feature_extractor(self) -> keras.Model:
        """Build CNN with attention mechanism for better feature extraction."""
        input_layer = keras.layers.Input(shape=(256, 256, 1))
        
        # Convolutional blocks with residual connections
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Residual block
        residual = x
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Add residual connection
        x = keras.layers.Add()([x, keras.layers.Conv2D(64, (1, 1))(residual)])
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # More conv layers
        x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Spatial attention
        attention = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = keras.layers.Multiply()([x, attention])
        
        # Global pooling and dense layers
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        features = keras.layers.Dense(128, name='feature_vector')(x)
        
        model = keras.Model(input_layer, features)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def _build_enhanced_similarity_model(self) -> keras.Model:
        """Enhanced similarity model with better architecture."""
        input_a = keras.layers.Input(shape=(128,), name='features_a')
        input_b = keras.layers.Input(shape=(128,), name='features_b')
        
        # Multiple comparison methods
        diff = keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([input_a, input_b])
        mult = keras.layers.Multiply()([input_a, input_b])
        concat = keras.layers.Concatenate()([input_a, input_b, diff, mult])
        
        # Attention mechanism for feature importance
        attention = keras.layers.Dense(512, activation='relu')(concat)
        attention = keras.layers.Dense(512, activation='sigmoid')(attention)
        attended_features = keras.layers.Multiply()([concat, attention])
        
        # Deep similarity computation
        x = keras.layers.Dense(256, activation='relu')(attended_features)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        
        similarity = keras.layers.Dense(1, activation='sigmoid', name='similarity')(x)
        
        model = keras.Model(inputs=[input_a, input_b], outputs=similarity)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _build_enhanced_quality_model(self) -> keras.Model:
        """Enhanced quality assessment with multi-scale analysis."""
        input_layer = keras.layers.Input(shape=(256, 256, 1))
        
        # Multi-scale feature extraction
        # Scale 1: Full resolution
        conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)
        
        # Scale 2: Half resolution
        conv2 = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(pool1)
        pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
        
        # Scale 3: Quarter resolution
        conv3 = keras.layers.Conv2D(64, (7, 7), activation='relu', padding='same')(pool2)
        pool3 = keras.layers.GlobalAveragePooling2D()(conv3)
        
        # Dense layers for quality prediction
        x = keras.layers.Dense(128, activation='relu')(pool3)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        quality = keras.layers.Dense(1, activation='sigmoid', name='quality_score')(x)
        
        model = keras.Model(input_layer, quality)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _build_enhanced_liveness_model(self) -> keras.Model:
        """Enhanced liveness detection with temporal analysis."""
        input_layer = keras.layers.Input(shape=(256, 256, 1))
        
        # Texture analysis branch
        texture_branch = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        texture_branch = keras.layers.MaxPooling2D((2, 2))(texture_branch)
        texture_branch = keras.layers.Conv2D(64, (3, 3), activation='relu')(texture_branch)
        texture_branch = keras.layers.GlobalAveragePooling2D()(texture_branch)
        
        # Edge analysis branch
        edge_branch = keras.layers.Conv2D(16, (1, 1), activation='relu')(input_layer)
        edge_branch = keras.layers.Conv2D(32, (3, 3), activation='relu')(edge_branch)
        edge_branch = keras.layers.Conv2D(64, (5, 5), activation='relu')(edge_branch)
        edge_branch = keras.layers.GlobalAveragePooling2D()(edge_branch)
        
        # Combine branches
        combined = keras.layers.Concatenate()([texture_branch, edge_branch])
        x = keras.layers.Dense(128, activation='relu')(combined)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        liveness = keras.layers.Dense(1, activation='sigmoid', name='liveness_score')(x)
        
        model = keras.Model(input_layer, liveness)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with adaptive enhancement."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        image = cv2.resize(image, (256, 256))
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add noise reduction
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Add dimensions
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using enhanced CNN."""
        processed_image = self.preprocess_image(image)
        features = self.feature_extractor.predict(processed_image, verbose=0)
        return features[0]
    
def assess_image_quality(self, image: np.ndarray) -> float:
        """Assess quality using enhanced model."""
        processed_image = self.preprocess_image(image)
        quality_score = self.quality_model.predict(processed_image, verbose=0)
        return float(quality_score[0][0])
    
    def detect_liveness(self, image: np.ndarray) -> Tuple[bool, float]:
        """Detect if fingerprint is from live finger."""
        processed_image = self.preprocess_image(image)
        liveness_score = self.liveness_model.predict(processed_image, verbose=0)
        score = float(liveness_score[0][0])
        is_live = score > 0.5
        return is_live, score
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity using enhanced neural model."""
        features1_reshaped = features1.reshape(1, -1)
        features2_reshaped = features2.reshape(1, -1)
        
        similarity_score = self.similarity_model.predict([features1_reshaped, features2_reshaped], verbose=0)
        return float(similarity_score[0][0])
    
    def match_fingerprints(self, image1: np.ndarray, image2: np.ndarray, 
                          threshold: float = 0.7) -> Tuple[bool, float, Dict]:
        """Comprehensive fingerprint matching with quality checks."""
        # Quality assessment
        quality1 = self.assess_image_quality(image1)
        quality2 = self.assess_image_quality(image2)
        
        # Liveness detection
        live1, liveness_score1 = self.detect_liveness(image1)
        live2, liveness_score2 = self.detect_liveness(image2)
        
        # Extract features
        features1 = self.extract_deep_features(image1)
        features2 = self.extract_deep_features(image2)
        
        # Calculate similarity
        similarity = self.calculate_similarity(features1, features2)
        
        # Decision logic with quality weighting
        min_quality = min(quality1, quality2)
        adjusted_threshold = threshold + (0.3 * (1 - min_quality))  # Higher threshold for low quality
        
        match = similarity > adjusted_threshold and live1 and live2 and min_quality > 0.3
        
        match_info = {
            'similarity': similarity,
            'quality1': quality1,
            'quality2': quality2,
            'liveness1': liveness_score1,
            'liveness2': liveness_score2,
            'is_live1': live1,
            'is_live2': live2,
            'adjusted_threshold': adjusted_threshold,
            'min_quality': min_quality
        }
        
        return match, similarity, match_info
    
    def save_models(self, path: str):
        """Save all trained models."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.feature_extractor.save(os.path.join(path, 'feature_extractor.h5'))
        self.similarity_model.save(os.path.join(path, 'similarity_model.h5'))
        self.quality_model.save(os.path.join(path, 'quality_model.h5'))
        self.liveness_model.save(os.path.join(path, 'liveness_model.h5'))
        
        # Save scaler
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_models(self, path: str):
        """Load all trained models."""
        self.feature_extractor = keras.models.load_model(os.path.join(path, 'feature_extractor.h5'))
        self.similarity_model = keras.models.load_model(os.path.join(path, 'similarity_model.h5'))
        self.quality_model = keras.models.load_model(os.path.join(path, 'quality_model.h5'))
        self.liveness_model = keras.models.load_model(os.path.join(path, 'liveness_model.h5'))
        
        # Load scaler
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
    
    def train_with_data(self, training_data: Dict, validation_data: Dict = None):
        """Train models with provided data."""
        images = training_data['images']
        features = training_data.get('features')
        quality_labels = training_data.get('quality_labels')
        liveness_labels = training_data.get('liveness_labels')
        similarity_pairs = training_data.get('similarity_pairs')
        similarity_labels = training_data.get('similarity_labels')
        
        # Train feature extractor if features are provided
        if features is not None:
            processed_images = np.array([self.preprocess_image(img)[0] for img in images])
            self.feature_extractor.fit(processed_images, features, 
                                     epochs=50, batch_size=32, verbose=1,
                                     validation_split=0.2 if validation_data is None else 0)
        
        # Train quality model
        if quality_labels is not None:
            processed_images = np.array([self.preprocess_image(img)[0] for img in images])
            self.quality_model.fit(processed_images, quality_labels,
                                 epochs=30, batch_size=32, verbose=1,
                                 validation_split=0.2 if validation_data is None else 0)
        
        # Train liveness model
        if liveness_labels is not None:
            processed_images = np.array([self.preprocess_image(img)[0] for img in images])
            self.liveness_model.fit(processed_images, liveness_labels,
                                  epochs=30, batch_size=32, verbose=1,
                                  validation_split=0.2 if validation_data is None else 0)
        
        # Train similarity model
        if similarity_pairs is not None and similarity_labels is not None:
            features_a = np.array([pair[0] for pair in similarity_pairs])
            features_b = np.array([pair[1] for pair in similarity_pairs])
            
            self.similarity_model.fit([features_a, features_b], similarity_labels,
                                    epochs=40, batch_size=64, verbose=1,
                                    validation_split=0.2 if validation_data is None else 0)


# Complete Intelligent Fingerprint System Integration
class CompleteFingerprintSystem:
    """Complete neural-enhanced fingerprint authentication system."""
    
    def __init__(self, db_path: str = "complete_fp_system.sqlite"):
        self.database = IntelligentFingerprintDatabase(db_path)
        self.processor = EnhancedFingerprintAIProcessor()
        self.session_data = {}
        
    def enroll_user(self, user_id: str, fingerprint_images: List[np.ndarray], 
                   metadata: Dict = None) -> bool:
        """Enroll user with multiple fingerprint samples."""
        if not fingerprint_images:
            return False
        
        # Process all samples and create consolidated template
        all_features = []
        quality_scores = []
        
        for image in fingerprint_images:
            # Quality check
            quality = self.processor.assess_image_quality(image)
            if quality < 0.4:  # Skip low quality samples
                continue
                
            # Liveness check
            is_live, liveness_score = self.processor.detect_liveness(image)
            if not is_live:
                continue
                
            # Extract features
            features = self.processor.extract_deep_features(image)
            all_features.append(features)
            quality_scores.append(quality)
        
        if not all_features:
            return False
        
        # Create consolidated template (weighted average by quality)
        features_array = np.array(all_features)
        weights = np.array(quality_scores)
        weights = weights / weights.sum()  # Normalize weights
        
        consolidated_template = np.average(features_array, axis=0, weights=weights)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'enrollment_samples': len(all_features),
            'average_quality': np.mean(quality_scores),
            'consolidation_method': 'weighted_average'
        })
        
        # Store in database
        self.database.store_template(user_id, consolidated_template, metadata)
        return True
    
    def authenticate_user(self, user_id: str, fingerprint_image: np.ndarray, 
                         confidence_threshold: float = 0.7) -> Dict:
        """Authenticate user with comprehensive analysis."""
        # Get stored template
        stored_data = self.database.get_template(user_id)
        if not stored_data:
            return {
                'authenticated': False,
                'error': 'User not found',
                'confidence': 0.0
            }
        
        # Process input image
        quality = self.processor.assess_image_quality(fingerprint_image)
        is_live, liveness_score = self.processor.detect_liveness(fingerprint_image)
        
        if quality < 0.3:
            return {
                'authenticated': False,
                'error': 'Image quality too low',
                'quality': quality,
                'confidence': 0.0
            }
        
        if not is_live:
            return {
                'authenticated': False,
                'error': 'Liveness detection failed',
                'liveness_score': liveness_score,
                'confidence': 0.0
            }
        
        # Extract features and compare
        query_features = self.processor.extract_deep_features(fingerprint_image)
        stored_features = stored_data['features']
        
        # Calculate similarity
        similarity = self.processor.calculate_similarity(query_features, stored_features)
        
        # Adaptive threshold based on template quality and usage
        template_quality = stored_data['quality_score']
        usage_count = stored_data['usage_count']
        
        # Lower threshold for high-quality, frequently used templates
        adaptive_threshold = confidence_threshold - (0.1 * template_quality) - (0.05 * min(usage_count / 100, 0.2))
        
        authenticated = similarity > adaptive_threshold
        
        # Update template if authentication successful
        if authenticated:
            self.database.adaptive_update_template(
                user_id, query_features, similarity, True
            )
        else:
            # Still log failed attempt for analysis
            self.database.adaptive_update_template(
                user_id, query_features, similarity, False
            )
        
        return {
            'authenticated': authenticated,
            'confidence': similarity,
            'quality': quality,
            'liveness_score': liveness_score,
            'adaptive_threshold': adaptive_threshold,
            'template_quality': template_quality,
            'usage_count': usage_count
        }
    
    def identify_user(self, fingerprint_image: np.ndarray, 
                     max_candidates: int = 10) -> List[Dict]:
        """Identify user from fingerprint (1:N matching)."""
        # Process input image
        quality = self.processor.assess_image_quality(fingerprint_image)
        is_live, liveness_score = self.processor.detect_liveness(fingerprint_image)
        
        if quality < 0.4 or not is_live:
            return []
        
        # Extract features
        query_features = self.processor.extract_deep_features(fingerprint_image)
        
        # Fast candidate retrieval using intelligent search
        candidates = self.database.intelligent_search(query_features, max_candidates)
        
        # Detailed matching for each candidate
        results = []
        for user_id, preliminary_similarity in candidates:
            # Get full authentication result
            auth_result = self.authenticate_user(user_id, fingerprint_image)
            
            if auth_result['authenticated']:
                results.append({
                    'user_id': user_id,
                    'confidence': auth_result['confidence'],
                    'quality': auth_result['quality'],
                    'template_quality': auth_result['template_quality']
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results
    
    def get_system_analytics(self) -> Dict:
        """Get comprehensive system analytics."""
        db_analytics = self.database.get_analytics()
        
        # Add system-specific metrics
        system_analytics = {
            'database_stats': db_analytics,
            'model_info': {
                'feature_extractor_params': self.processor.feature_extractor.count_params(),
                'similarity_model_params': self.processor.similarity_model.count_params(),
                'quality_model_params': self.processor.quality_model.count_params(),
                'liveness_model_params': self.processor.liveness_model.count_params()
            }
        }
        
        return system_analytics
    
    # Perform system maintenance and optimization which invlves cleaning the discarded attemps 
    # for the model to learn the 
    def cleanup_and_optimize(self):
        # Clean up low quality templates
        removed_count = self.database.cleanup_low_quality_templates()
        
        # Could add more optimization routines here
        # - Retrain models with recent data
        # - Update clustering
        # - Compress old authentication logs
        
        return {
            'low_quality_templates_removed': removed_count,
            'optimization_completed': True
        }