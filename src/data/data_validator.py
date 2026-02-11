from pydantic import BaseModel,Field, field_validator
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

class CreditRiskInput(BaseModel):
    """Input validation schema for API"""
    age: int = Field(..., gt=18, lt=100)
    income: float = Field(..., gt=0)
    debt: float = Field(..., ge=0)
    credit_limit: float = Field(..., gt=0)
    credit_used: float = Field(..., ge=0)
    employment_years: int = Field(..., ge=0)

    @field_validator('debt')
    @classmethod
    def validate_debt(cls,v,info):
        income = info.data.get("income")
        if 'income' and v >income * 5:
            raise ValueError('Debt cannot exceed 5x income')
        return v
    
    @field_validator('credit_used')
    @classmethod
    def validate_credit_usage(cls,v,info):
        limit = info.data.get("credit_limit")
        if limit and v>limit:
            raise ValueError("Credit used cannot exceed credit limit ")
        return v
    
    model_config ={
        "json_schema_extra" : {
            "example": {
                "age": 35,
                "income": 50000,
                "debt": 5000,
                "credit_limit": 25000,
                "credit_used": 3000,
                "employment_years": 5
            }
        }
    }


"""Data Frame Validator"""
class DataValidator:
    """Validate dataframe"""

    REQUIRED_COLUMNS: List[str] = [
        "age",
        "income",
        "debt",
        "credit_limit",
        "credit_used",
        "employment_years"
    ]

    def validate(self,df:pd.DataFrame)->None:
        logger.info("Staring Data Validation")

        self._check_columns(df)
        self._check_nulls(df)
        self._check_income(df)
        self._check_credit_usage(df)

        logger.info("Data Validation completed successfully")

    def _check_columns(self,df:pd.DataFrame)->None:
        missing_cols = list(set(self.REQUIRED_COLUMNS)-set(df.columns))
        if missing_cols:
            logger.error(f"Missing columns:{missing_cols}")
            raise ValueError(f"Missing columns {missing_cols}")
        
    def _check_nulls(self,df:pd.DataFrame)->None:
        nulls_counts = df[self.REQUIRED_COLUMNS].isnull().sum()
        if nulls_counts.sum()>0:
            logger.warning(f"Null values detected:\n {nulls_counts[nulls_counts>0]}")

    def _check_age(self, df: pd.DataFrame) -> None:
        invalid_age = df[(df["age"] < 18) | (df["age"] > 100)]
        if not invalid_age.empty:
            logger.error(
                f"Invalid age values found: {len(invalid_age)} rows "
                f"(age must be between 18 and 100)"
            )
            raise ValueError("Invalid age values in dataset")

    
    def _check_income(self, df: pd.DataFrame) -> None:
        invalid_income = df[df["income"] <= 0]
        if not invalid_income.empty:
            logger.error(
                f"Invalid income values found: {len(invalid_income)} rows "
                f"(income must be > 0)"
            )
            raise ValueError("Invalid income values in dataset")

    def _check_credit_usage(self, df: pd.DataFrame) -> None:
        invalid_credit = df[df["credit_used"] > df["credit_limit"]]
        if not invalid_credit.empty:
            logger.error(
                f"Credit used exceeds limit in {len(invalid_credit)} rows"
            )
            raise ValueError("Credit usage exceeds credit limit")



    
        
    
if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    logger = setup_logger('data_validator')

    validator = DataValidator()
    df = pd.read_csv('data/raw/credit_risk.csv')
    validator.validate(df)
    print("Validation passed!")

