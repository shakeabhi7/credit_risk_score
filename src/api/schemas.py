from pydantic import BaseModel, Field, model_validator
from typing import Optional

class CreditRiskInput(BaseModel):
    """Input validation for credit risk prediction"""
    age: int = Field(...,gt=18,lt=100, description="Customer age (18-99)")
    income:float = Field(...,gt=0,description="Annual income in USD")
    debt: float = Field(...,ge=0,description="Total debt in USD")
    credit_limit:float = Field(...,gt=0,description="Credit card limit in USD")
    credit_used: float = Field(..., ge=0, description="Credit used in USD")
    employment_years: int = Field(..., ge=0, description="Years of employment (0+)")
    employment_type: str = Field(..., description="Employment type (Salaried/Self-employed/etc)")

    # Custom validation
    @model_validator(mode="after")
    def check_credit_usage(self):
        if self.credit_used > self.credit_limit:
            raise ValueError("credit_used cannot be greater than credit_limit")
        return self


    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 30,
                "income": 75000,
                "debt": 15000,
                "credit_limit": 50000,
                "credit_used": 20000,
                "employment_years": 3,
                "employment_type": "Salaried"
            }
        }
    }

class PredictionResponse(BaseModel):
    """Response format for predictions"""

    customer_id: str = Field(..., description="Auto-generated customer ID")
    predicted_class: int = Field(..., description="0=Good Credit, 1=Bad Credit")
    probability: float = Field(..., ge=0, le=1, description="Probability of prediction (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    model_name: str = Field(..., description="Name of model used")
    message: str = Field(..., description="Human-readable prediction message")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST_1708017930_a7f3c9e1",
                "predicted_class": 0,
                "probability": 0.87,
                "confidence": 0.92,
                "model_name": "XGBoost",
                "message": "Good credit customer - Low risk"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error responce format"""

    error: str = Field(...,description="Error message")
    status_code:int = Field(...,description="HTTP status code")
    detail:Optional[str] = Field(None,description="Additional error details")

class HealthResponse(BaseModel):
    """Health check Response"""

    status: str = Field(...,description="API status")
    version:str = Field(...,description="API version")
    models_loaded :bool = Field(...,description="Are models loaded?")
    database_connected:bool = Field(...,description="Is Database connected?")