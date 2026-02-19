import logging
from datetime import datetime
from pymongo import MongoClient
import yaml
import os

logger = logging.getLogger(__name__)

class PredictionLogger:
    """Logger predictions to MongoDB-using YAML config"""

    def __init__(self,config_path = 'src/config/data_config.yaml'):
        """Initialize with config from YAML"""
        self.config_path = config_path
        self.client = None
        self.db = None
        self.connected = False

        try:
            # Load config from YAML

            with open(config_path,'r') as f:
                config = yaml.safe_load(f)

            mongodb_config = config.get("mongodb", {})

            self.url = mongodb_config.get("url")
            self.db_name = mongodb_config.get("database")
            self.collections = mongodb_config.get("collections", {})
            logger.info(f" Config loaded from {config_path}")
            logger.info(f"   Database: {self.db_name}")
            logger.info(f"   Collections: {list(self.collections.keys())}")
        
        except Exception as e:
            logger.error(f" Error loading config: {e}")
            raise

        # Connect to MongoDB
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB"""

        try:
            self.client = MongoClient(
                self.url,
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.connected = True
            logger.info(" Connected to MongoDB")

        except Exception as e:
            logger.warning(f" MongoDB connection failed: {e}")
            self.connected = False

    def _get_collection(self, key):
        """Get collection using YAML config"""

        collection_name = self.collections.get(key)

        if not collection_name:
            raise ValueError(f"Collection '{key}' not found in YAML")

        return self.db[collection_name]
    
    def log_training_session(self, session_data):
        """
        Log training session to training_sessions collection
        Uses config from YAML
        """
        if not self.connected:
            logger.warning(" MongoDB not connected, skipping logging")
            return None
        
        try:

            collection = self._get_collection("training_sessions")

            record = {
                "metadata": session_data.get("metadata", {
                    "timestamp": datetime.utcnow()
                }),

                "input_data": session_data.get("input_data", {}),

                "engineered_features": session_data.get("engineered_features", {}),

                "preprocessing_info": session_data.get("preprocessing_info", {}),

                "prediction": session_data.get("prediction", {})
            }

            result = collection.insert_one(record)

            logger.info(f" Training session saved: {result.inserted_id}")
            logger.info(f"   Stored fields: metadata, input_data, engineered_features, preprocessing_info, prediction")

            return result.inserted_id
        except Exception as e:
            logger.error(f" Error logging training session: {e}")
            return None
    
    def log_inference(self, inference_data):
        """
        Log inference/prediction to predictions collection
        Uses config from YAML
        
        inference_data must have exactly 5 fields:
        - metadata
        - input_data
        - engineered_features
        - preprocessing_info
        - prediction
        """
        if not self.connected:
            logger.warning(" MongoDB not connected, skipping logging")
            return None
        try:

            collection = self._get_collection("predictions")

            record = {
                "metadata": inference_data.get("metadata", {
                    "timestamp": datetime.utcnow()
                }),

                "input_data": inference_data.get("input_data", {}),

                "engineered_features": inference_data.get("engineered_features", {}),

                "preprocessing_info": inference_data.get("preprocessing_info", {}),

                "prediction": inference_data.get("prediction", {})
            }

            result = collection.insert_one(record)
            logger.info(f" Inference saved: {result.inserted_id}")
            logger.info(f"   Stored fields: metadata, input_data, engineered_features, preprocessing_info, prediction")

            return result.inserted_id

        except Exception as e:
            logger.error(f" Inference log failed: {e}")
            return None
        
    def log_api_call(self, api_data):
        """
        Log API call to api_logs collection
        Uses config from YAML
        """
        if not self.connected:
            logger.warning(" MongoDB not connected, skipping logging")
            return None
        
        try:

            collection = self._get_collection("api_logs")

            record = {
                "timestamp": api_data.get("timestamp", datetime.utcnow()),
                "endpoint": api_data.get("endpoint"),
                "method": api_data.get("method"),
                "status_code": api_data.get("status_code"),
                "response_time_ms": api_data.get("response_time_ms"),
                "user_id": api_data.get("user_id")
            }

            result = collection.insert_one(record)

            logger.info(f" API log saved: {result.inserted_id}")

            return result.inserted_id

        except Exception as e:
            logger.error(f" API log failed: {e}")
            return None
        
    def get_training_sessions(self, limit=10):
        """Get training session history"""
        if not self.connected:
            logger.warning(" MongoDB not connected")
            return []
        try:

            collection = self._get_collection("training_sessions")

            sessions = list(
                collection.find()
                .sort("metadata.timestamp", -1)
                .limit(limit)
            )

            return sessions

        except Exception as e:
            logger.error(e)
            return []
        
    def get_predictions_by_model(self, model_name, limit=100):
        """Get predictions by specific model"""
        if not self.connected:
            logger.warning(" MongoDB not connected")
            return []
        try:

            collection = self._get_collection("predictions")

            predictions = list(
                collection.find(
                    {"prediction.model_name": model_name}
                ).limit(limit)
            )

            return predictions

        except Exception as e:
            logger.error(e)
            return []


    def close(self):

        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    
    logger = setup_logger('prediction_logger')

    # Initialize logger with config
    pred_logger = PredictionLogger(config_path='src/config/data_config.yaml')
    if pred_logger.connected:
        session_data = {
            'metadata': {
                'timestamp': datetime.utcnow(),
                'model_version': 'v1.0',
                'api_version': 'v1.0'
            },
            'input_data': {
                'training_samples': 3800,
                'test_samples': 950,
                'features_count': 15
            },
            'engineered_features': {
                'features_list': ['debt_to_income', 'credit_utilization', 'income_per_age']
            },
            'preprocessing_info': {
                'scaler_type': 'StandardScaler',
                'encoder_type': 'LabelEncoder'
            },
            'prediction': {
                'model_name': 'XGBoost',
                'accuracy': 0.85,
                'auc_roc': 0.87,
                'f1_score': 0.80,
                'precision': 0.82,
                'recall': 0.78
            }
        }
        
        session_id = pred_logger.log_training_session(session_data)
        logger.info(f" Sample training session logged: {session_id}")
    else:
        logger.warning(" Skipping sample logging (MongoDB not available)")
    
    pred_logger.close()