from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from src.monitoring.logger import setup_logger
from src.api.models_loader import ModelLoader
from src.api.inference_service import InferenceService
from src.api.schemas import CreditRiskInput, PredictionResponse, HealthResponse
from src.database.prediction_logger import PredictionLogger

logger = setup_logger('fastapi_main')

# CONFIG
API_VERSION = "v1.0"
MODEL_VERISON = "v1.0"

# Lifespan 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("-"*6)
    logger.info("FASTAPI Startup")
    logger.info("-"*6)

    try:
        #load models
        logger.info("Loading models..")
        app.state.models_loader = ModelLoader()

        success = app.state.models_loader.load_all(best_model_name="xgboost")
        if not success:
            raise Exception("Failed to load models")
        
        # create inference service
        app.state.inference_service = InferenceService(
            app.state.models_loader,
            app.state.models_loader.get_preprocessor(),
            app.state.models_loader.get_feature_engineer(),
        )
        logger.info("Inference service created")

        #Database logger
        app.state.db_logger = PredictionLogger()

        if app.state.db_logger.connected:
            logger.info("Database connected")
        else:
            logger.warning("Database not available")
        
        logger.info("API Startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}",exc_info=True)
        raise
    yield

    # Shutdown logic
    logger.info("-"*6)
    logger.info("FASTAPI Shutdown")
    logger.info("-"*6)

    if app.state.db_logger:
        app.state.db_logger.close()
    logger.info("API SHUTDOWN COMPLETE")
# FASTAPI APP

app = FastAPI(
    title = "Credit Risk Scoring API",
    description="Reali-time credit risk prediction API",
    version=API_VERSION,
    lifespan=lifespan,
)

# Health check

@app.get("/health",response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    models_loader = app.state.models_loader
    db_logger= app.state.db_logger

    return {
        "status": "healthy",
        "version" : API_VERSION,
        "models_loaded":models_loader.is_loaded if models_loader else False,
        "database_connected": db_logger.connected if db_logger else False,
    }

# prediction endpoint
@app.post("/predict",response_model=PredictionResponse)
async def predict(request: CreditRiskInput):
    logger.info("-"*6)
    logger.info("Prediction request received")
    logger.info("-"*6)

    try:
        models_loader = app.state.models_loader
        inference_service = app.state.inference_service
        db_logger = app.state.db_logger

        if not models_loader or not models_loader.is_loaded:
            raise HTTPException(status_code=503,detail="Models not loaded")
        if not inference_service:
            raise HTTPException(status_code=503,detail="Inference service not ready")
        
        # Run inference

        result = inference_service.run_inference(request.model_dump())

        # Database logging
        if db_logger and db_logger.connected:
            try:
                inference_data = {
                    "metadata":{
                        "customer_id":result["customer_id"],
                        "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")),
                        "model_version":MODEL_VERISON,
                        "api_version":API_VERSION,
                    },
                    "input_data": result["input_data"],
                    "engineered_features": result["engineered_features"],
                    "preprocessing_info": {
                        "scaler_type": "StandardScaler",
                        "features_used": list(request.model_dump().keys()),
                    },
                    "prediction": {
                        "predicted_class": result["predicted_class"],
                        "probability": result["probability"],
                        "confidence": result["confidence"],
                        "model_name": result["model_name"],
                    },
                }
                db_logger.log_inference(inference_data)

                logger.info("Prediction logged to database")

            except Exception as e:
                logger.warning(f"Failed to log to database:{e}")

        response = {
                "customer_id": result["customer_id"],
                "predicted_class": result["predicted_class"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "model_name": result["model_name"],
                "message": result["message"],
            }
        logger.info("Prediction response sent")

        return response
    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message":"Credit Risk Scoring API",
        "version": API_VERSION,
        "endpoints": {
            "health":"/health",
            "predict":"/predict",
            "docs":"/docs",
        },
    }

# Run Server

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server..")

    uvicorn.run(
        "src.api.main.api",
        host = "0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

