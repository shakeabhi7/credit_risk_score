import joblib
import os
import yaml
from sklearn.model_selection import train_test_split
import logging
from src.preprocessing.preprocessor import Preprocessor
from src.features.feature_engineer import FeatureEngineer
from src.data.data_loader import DataLoader

logger = logging.getLogger(__name__)

class DataPipeline:
    """Complete data preprocessing pipeline"""

    def __init__(self,
                data_config_path="src/config/data_config.yaml"):
        
        self.data_config = self._load_yaml(data_config_path)

        self.test_size = self.data_config["data"]["test_size"]
        self.random_state = self.data_config["data"]["random_state"]

        self.processed_path = self.data_config["processed_data_path"]
        self.preprocessor = None
        self.feature_engineer = None

        logger.info("Configuration loaded successfully")
    
    def _load_yaml(self,path):
        with open(path,"r") as file:
            return yaml.safe_load(file)
    
    def run(self,df):
        """
        Complete pipeline:
        1. Feature engineering
        2. Train-test split
        3. Preprocessing (scaling + encoding)
        4. Save artifacts
        """
         
        logger.info("-"*6)
        logger.info("STARTING DATA PIPELINE")
        logger.info("-"*6)

        # Step 1: Feature Engineering
        logger.info("Step 1: Feature Engineering")
        logger.info("-"*6)
        self.feature_engineer = FeatureEngineer()
        df = self.feature_engineer.create_features(df)

        #Step 2: Drop ID column
        logger.info("Step 2: Dropping customer_id column")
        if "customer_id" in df.columns:
            df = df.drop(columns=["customer_id"])
            logger.info("Dropped customer_id")

        # Step 3: Separate features and target
        logger.info("Step 3: Separating features and target")
        logger.info("-"*6)
        X = df.drop('target', axis=1)
        y = df['target']
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")

        # Step 4: Train-test split
        logger.info("\nStep 4: Train-test split (80-20)")
        logger.info("-"*6)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Training target distribution:")
        logger.info(f"  Good (0): {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
        logger.info(f"  Bad (1): {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        
        # Step 5: Preprocessing (fit on train, transform train+test)
        logger.info("Step 5: Preprocessing (scaling + encoding)")
        logger.info("-"*6)
        self.preprocessor = Preprocessor()
        self.preprocessor.fit(X_train)

        X_train_processed = self.preprocessor.transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        logger.info(f"Training data scaled: {X_train_processed.shape}")
        logger.info(f"Test data scaled: {X_test_processed.shape}")

        # Step 6: Save artifacts
        logger.info("Step 6: Saving artifacts")
        logger.info("-"*6)

        self._save_artifacts(X_train_processed,X_test_processed,y_train,y_test)

        logger.info("-"*6)
        logger.info("PIPELINE COMPLETE!")
        logger.info("-"*0)

        return X_train_processed, X_test_processed, y_train, y_test
    
    def _save_artifacts(self,X_train,X_test,y_train,y_test):
        os.makedirs(self.processed_path,exist_ok=True)

        #save datasets
        X_train.to_csv(os.path.join(self.processed_path,"X_train.csv"),index=False)
        X_test.to_csv(os.path.join(self.processed_path,"X_test.csv"),index=False)
        y_train.to_csv(os.path.join(self.processed_path,"y_train.csv"),index=False)
        y_test.to_csv(os.path.join(self.processed_path,"y_test.csv"),index=False)

        logger.info("Saved processed datasets")

        #Save preprocessor
        self.preprocessor.save(os.path.join(self.processed_path,"preprocessor.joblib"))
        logger.info("Saved: preprocessor.joblib")

        # save feature engineer
        joblib.dump(self.feature_engineer,
                    os.path.join(self.processed_path,"feature_engineer.joblib"))
        logger.info("Saved: feature_engineer.joblib")
      
        #save metadata
        info = {
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "features": X_train.columns.tolist(),
            "engineered_features": self.feature_engineer.get_feature_names(),
            "train_test_split": f"{int((1-self.test_size)*100)}-{int(self.test_size*100)}",
        }
        joblib.dump(info,
                    os.path.join(self.processed_path,"data_info.joblib"))
        
        logger.info("Saved: data_info.joblib")


if __name__ == "__main__":
    from src.monitoring.logger import setup_logger

    logger = setup_logger("data_pipeline")

    loader = DataLoader()
    df = loader.load_csv("data/processed/credit_risk_cleaned.csv")

    pipeline = DataPipeline()
    X_train,X_test,y_train,y_test = pipeline.run(df)

    logger.info("Final shapes:")
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_test: {X_test.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"y_test: {y_test.shape}")




        
