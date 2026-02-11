import pandas as pd
import logging
from pymongo import MongoClient
import yaml

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading from CSV or MongoDB"""
    def __init__(self,config_path = 'src/config/data_config.yaml'):
        with open(config_path,'r') as f:
            self.config = yaml.safe_load(f)
        self.mongo_client = None
    
    def load_csv(self,filepath):
        """load csv"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f'CSV loaded: {filepath},shape:{df.shape}')
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def connect_mongodb(self):
        """Connect to MongoDB"""
        try:
            mongo_url = self.config['mongodb']['url']
            self.mongo_client = MongoClient(mongo_url,serverSelectTimeoutMS = 5000)
            self.mongo_client.admin.command('ismaster')
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.warning(f"MongoDB not available:{e}")
            logger.warning("Continuing without MongoDB")

    def load_from_mongodb(self,collection_name,query=None):
        """Load from MongoDB collection"""
        if not self.mongo_client:
            self.connect_mongodb()
        try:
            db = self.mongo_client[self.config['mongodb']['database']]
            collection = db[collection_name]

            if query is None:
                query = {}
            
            data = list(collection.find(query))
            df = pd.DataFrame(data)

            if '_id' in df.columns:
                df = df.drop('_id',axis=1)
            
            logger.info(f"✅ Loaded from MongoDB: {collection_name}, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading from MongoDB:{e}")
            raise

    def close(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB Connection closed")

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    logger = setup_logger('data_loader')

    loader = DataLoader()
    df = loader.load_csv('data/raw/credit_risk.csv')
    print(f'\nData loaded successfully!')
    print(f"Shape : {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
                         
