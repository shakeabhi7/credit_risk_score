from pymongo import MongoClient
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBClient:
    """MongoDB operations"""
    
    def __init__(self, connection_string='mongodb://localhost:27017/'):
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ismaster')
            self.db = self.client['credit_risk_db']
            logger.info("MongoDB client initialized")
            self.connected = True
        except Exception as e:
            logger.warning(f"MongoDB not available: {e}")
            self.client = None
            self.db = None
            self.connected = False
    
    def insert_prediction(self, record):
        """Insert prediction record"""
        if not self.connected:
            logger.warning("MongoDB not connected, skipping storage")
            return None
        
        try:
            collection = self.db['credit_risk_predictions']
            record['created_at'] = datetime.utcnow()
            result = collection.insert_one(record)
            logger.info(f"Prediction stored with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            return None
    
    def get_predictions(self, query=None, limit=100):
        """Retrieve predictions"""
        if not self.connected:
            logger.warning("MongoDB not connected")
            return []
        
        try:
            collection = self.db['credit_risk_predictions']
            if query is None:
                query = {}
            results = list(collection.find(query).limit(limit))
            logger.info(f"Retrieved {len(results)} predictions")
            return results
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            return []
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    logger = setup_logger('mongo_client')
    
    db = MongoDBClient()
    if db.connected:
        print("MongoDB connected successfully!")
    else:
        print("MongoDB not available (you can continue without it)")