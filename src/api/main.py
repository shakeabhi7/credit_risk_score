from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
import sys
import os
from datetime import datetime

from src.monitoring.logger import setup_logger
from src.api.models_loader import ModelsLoader
from src.api.inference_service import InferenceService
from src.api.schemas import CreditRiskInput, PredictionResponse, HealthResponse
from src.database.prediction_logger import PredictionLogger

logger = setup_logger('fastapi_main')

# initialize FASTAPI

app = FastAPI(
    title = "Credit Risk Scoring API",
    description="Reali-time credit risk prediction API",
    version="1.0.0"
)

# Global state

models_loader = None
inference_service = None
db_logger = None
api_version = "v1.0"
model_version = "v1.0"

#Startup & shutdown

