import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create engineered features for credit risk prediction"""

    def __init__(self):
        self.created_features = []
    
    def create_features(self,df):
        """
        Create engineered features:
        1. debt_to_income
        2. credit_utilization
        3. income_per_age
        4. employment_stability
        5. income_to_credit
        6. income_squared
        7. debt_impact
        8. age_group (categorical)
        """
        logger.info("Creating engineered features..")
        df = df.copy()

        # Debt to Income Ratio
        df['debt_to_income'] = df['debt']/(df['income']+1)
        logger.info("Created: debt_to_income")

        # Credit Utilization
        df['credit_utilization'] = df['credit_used']/(df['credit_limit']+1)
        logger.info("Created: credit_utilization")

        # Income per age
        df['income_per_age'] = df['income']/(df['age']+1)
        logger.info("Created: income_per_age")

        #Employment Stability (years employed/current age)
        df['employment_stability'] = df['employment_years']/(df['age']+1)
        logger.info("Created: employment_stability")

        #Income to credit Limit
        df['income_to_credit'] = df['income']/(df['credit_limit']+1)
        logger.info("Created: income_to_credit")

        #income squared 
        df['income_squared'] = df['income'] ** 2
        logger.info("Created: income_squared")

        # debt impact (debt * utilization)
        df['debt_impact'] = df['debt'] * df['credit_utilization']
        logger.info("Created: debt_impact")

        # Age Group(categorical)
        df['age_group'] = pd.cut(df['age'],
                                    bins=[0,25,35,45,55,100],
                                  labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        logger.info("Created: age_group")

        self.created_features = [
            'debt_to_income', 'credit_utilization', 'income_per_age',
            'employment_stability', 'income_to_credit', 'income_squared',
            'debt_impact', 'age_group'
        ]
        logger.info(f"Created: {len(self.created_features)} engineered features")

        return df
    
    def get_feature_names(self):
        """Get list of engineered feature names"""

        return self.created_features
    
    def save_engineered_features(self,df,path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        df.to_csv(path,index=False)
        logger.info(f"Engineered data saved at {path}") 

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    from src.data.data_loader import DataLoader

    logger = setup_logger('feature_engineer')

    loader = DataLoader()
    df = loader.load_csv('data/processed/credit_risk_cleaned.csv')

    engineer = FeatureEngineer()
    df_engineered = engineer.create_features(df)

    engineer.save_engineered_features(df_engineered,'data/processed/credit_risk_featured.csv')


    logger.info(f"Original columns: {len(df.columns)}")
    logger.info(f"After engineering: {len(df_engineered.columns)}")
    logger.info(f"New features:\n{df_engineered[engineer.get_feature_names()].head()}")
    logger.info(f"Full DataFrame:\n{df_engineered.head(5)}")