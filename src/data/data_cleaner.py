import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess data"""
    
    def __init__(self):
        self.original_shape = None
        self.cleaning_report = {}
    
    def clean(self, df):
        """
        Data Cleaning

        """
        logger.info("Starting data cleaning...")
        self.original_shape = df.shape
        df = df.copy()
        
        # Step 1: Remove duplicates
        logger.info("Step 1: Removing duplicates...")
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        duplicates_removed = duplicates_before - df.duplicated().sum()
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        logger.info(f" Removed {duplicates_removed} duplicate rows")
        
        # Step 2: Handle missing values
        logger.info("Step 2: Handling missing values...")
        missing_before = df.isnull().sum().sum()
        
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"   Filled {col} nulls with median: {median_val}")
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"   Filled {col} nulls with mode: {mode_val}")
        
        missing_after = df.isnull().sum().sum()
        self.cleaning_report['missing_values_filled'] = missing_before - missing_after
        logger.info(f" Filled {missing_before - missing_after} missing values")
        
        # Step 3: Validate data ranges
        logger.info("Step 3: Validating data ranges...")
        invalid_age = (df['age'] < 18) | (df['age'] > 100)
        invalid_income = df['income'] < 0
        invalid_debt = df['debt'] < 0
        invalid_employment = df['employment_years'] < 0
        
        invalid_mask = invalid_age | invalid_income | invalid_debt | invalid_employment
        invalid_count = invalid_mask.sum()
                
        if invalid_count > 0:
            logger.warning(f" Found {invalid_count} invalid values")
            df = df[~invalid_count]
            logger.info(f" Removed {invalid_count} rows with invalid ranges")
        
        self.cleaning_report['invalid_ranges_removed'] = invalid_count
        
        # Step 4: Handle outliers (IQR method)
        logger.info("Step 4: Handling outliers...")
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col != 'target'
        ]
        
        outliers_removed = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            removed = (~mask).sum()
            
            if removed > 0:
                df = df[mask]
                outliers_removed += removed
                logger.info(f"{col}: Removed {removed} outliers")
        
        self.cleaning_report['outliers_removed'] = outliers_removed
        logger.info(f" Removed {outliers_removed} outlier rows")
        
        # Generate report
        logger.info("\n" + "="*6)
        logger.info("DATA CLEANING REPORT")
        logger.info("="*6)
        logger.info(f"Original shape: {self.original_shape}")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Rows removed: {self.original_shape[0] - df.shape[0]}")
        logger.info(f" - Duplicates: {self.cleaning_report['duplicates_removed']}")
        logger.info(f" - Invalid ranges: {self.cleaning_report['invalid_ranges_removed']}")
        logger.info(f" - Outliers: {self.cleaning_report['outliers_removed']}")
        logger.info(f"Missing values filled: {self.cleaning_report['missing_values_filled']}")
        logger.info("="*6)
        
        return df

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    from src.data.data_loader import DataLoader
    
    logger = setup_logger('data_cleaner')
    
    loader = DataLoader()
    df = loader.load_csv('data/raw/credit_risk.csv')
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    
    # Save cleaned data
    df_clean.to_csv('data/processed/credit_risk_cleaned.csv', index=False)
    logger.info(" Cleaned data saved to data/processed/credit_risk_cleaned.csv")