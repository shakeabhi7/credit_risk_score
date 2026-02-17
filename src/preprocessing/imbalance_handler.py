import pandas as pd
from imblearn.over_sampling import SMOTE
import logging
import yaml
import os
logger = logging.getLogger(__name__)

class ImbalanceHandler:
    """Handle class imbalance using SMOTE"""

    def __init__(self,data_config_path="src/config/data_config.yaml"):
        self.data_config = self._load_yaml(data_config_path)
        
        self.random_state = self.data_config["data"]["random_state"]
        self.processed_path = self.data_config["processed_data_path"]
        self.files = self.data_config["files"]

        self.smote = SMOTE(random_state=self.random_state)
        self.isfitted = False

    def _load_yaml(self,path):
        with open(path,"r") as file:
            return yaml.safe_load(file)
        
    def fit_resample(self,X_train,y_train):
        """
        Apply SMOTE to training data
        Oversampling minority class 
        """
        logger.info("Handling class imbalance with SMOTE...")
        logger.info(f"Before SMOTE:")
        logger.info(f"  Good (0): {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
        logger.info(f"  Bad (1): {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        logger.info(f"  Ratio: {(y_train==0).sum() / (y_train==1).sum():.2f}:1")

        try:
            X_train_resampled,y_train_resampled = self.smote.fit_resample(X_train,y_train)
            logger.info(f" After SMOTE:")
            logger.info(f"  Good (0): {(y_train_resampled==0).sum()} ({(y_train_resampled==0).sum()/len(y_train_resampled)*100:.1f}%)")
            logger.info(f"  Bad (1): {(y_train_resampled==1).sum()} ({(y_train_resampled==1).sum()/len(y_train_resampled)*100:.1f}%)")
            logger.info(f"  Ratio: {(y_train_resampled==0).sum() / (y_train_resampled==1).sum():.2f}:1")
            logger.info(f"  SMOTE applied successfully!")
            logger.info(f"   Original training size: {len(X_train)}")
            logger.info(f"   Resampled training size: {len(X_train_resampled)}")

            self.isfitted = True
            return X_train_resampled, y_train_resampled
        except Exception as e:
            logger.error(f"Error applying SMOTE:{e}")
            raise

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    logger = setup_logger('imbalance_handler')

    handler = ImbalanceHandler()

    
    # Load paths from YAML
    X_train_path = os.path.join(handler.processed_path, handler.files["X_train"])
    y_train_path = os.path.join(handler.processed_path, handler.files["y_train"])

    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()

    # Apply SMOTE
    X_train_balanced, y_train_balanced = handler.fit_resample(X_train, y_train)

    #save balanced data using YAML paths
    X_bal_path = os.path.join(handler.processed_path,handler.files["X_train_balanced"])
    y_bal_path = os.path.join(handler.processed_path,handler.files["y_train_balanced"])

    X_train_balanced.to_csv(X_bal_path, index=False)
    y_train_balanced.to_csv(y_bal_path, index=False)

    logger.info("Balanced Data Saved:")
    logger.info(X_bal_path)
    logger.info(y_bal_path)

