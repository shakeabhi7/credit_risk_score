import pandas as pd
import numpy as np
import joblib
import logging
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        self.pipeline = None
        self.features_names = None
        self.is_fitted = None
    
    def fit(self,X,y=None):
        X = X.copy()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num',numeric_transformer,numeric_cols),
                ('cat',categorical_transformer,categorical_cols)
            ]
        )

        self.pipeline.fit(X)
        self.is_fitted = True

        #save final features names
        ohe_features = []
        if categorical_cols:
            ohe_features = (
                self.pipeline.named_transformers_['cat']
                .get_feature_names_out(categorical_cols)
                .tolist()
            ) 
        self.features_names = numeric_cols + ohe_features

        logger.info(f"Preprocessor fitted with {len(self.features_names)} features")

        return self
    def transform(self,X):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")

        X_transformed = self.pipeline.transform(X)

        return pd.DataFrame(
            X_transformed,
            columns=self.features_names
        )
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
    def save(self,filepath):
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        joblib.dump(self,filepath)
        logger.info(f"Preprocessor saved at {filepath}")
    
    @staticmethod
    def load(filepath):
        return joblib.load(filepath)
    
if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    from src.data.data_loader import DataLoader

    logger = setup_logger('preprocessor')

    loader = DataLoader()
    df = loader.load_csv('data/processed/credit_risk_featured.csv')

    X = df.drop('target',axis=1)
    pre = Preprocessor()
    X_processed = pre.fit_transform(X)

    logger.info(f"Original shape: {X.shape}")
    logger.info(f"Processed shape: {X_processed.shape}")
        