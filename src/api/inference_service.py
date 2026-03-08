import pandas as pd
import logging
import time
from uuid import uuid4

logger = logging.getLogger(__name__)

class InferenceService:
    """Handle predictions and inference logic"""

    def __init__(self,models_loader,preproccessor,feature_engineer):
        self.models_loader = models_loader
        self.preprocessor = preproccessor
        self.feature_engineer = feature_engineer
        self.best_model = models_loader.get_best_model()
        self.model_name = models_loader.best_model_name

    def prepare_input(self,input_data):
        """convert input to DataFrame"""
        logger.info("Preparing input data")

        try:
            # create dataframe from input
            df = pd.DataFrame([input_data])
            logger.info(f"Input prepared: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error preparing input: {e}")
            raise

    def preprocess(self,X):
        """Preprocess features"""
        logger.info("Preprocessing data..")
        try:
            X_processed = self.preprocessor.transform(X)
            logger.info(f"Preprocessing complete: {X_processed.shape}")
            return X_processed
        except Exception as e:
            logger.info(f"Error preprocessing: {e}")
            raise

    def engineer_features(self,X):
        """Engineer features"""
        logger.info("Engineering features..")
        try:
            X_engineered = self.feature_engineer.create_features(X)
            logger.info(f"Feature engineering complete:{X_engineered.shape}")
            return X_engineered
        except Exception as e:
            logger.info(f"Error engineering features: {e}")
            raise


    def predict(self,X):
        """Make prediction"""
        logger.info("Making prediction..")

        try:
            # Get prediction
            prediction = self.best_model.predict(X)[0]
            probability = self.best_model.predict_proba(X)[0][1]

            #calculate confidence
            confidence = max(self.best_model.predict_proba(X)[0])

            logger.info(f"Predictioon: {prediction}, Probability: {probability}")

            return {
                'prediction' : int(prediction),
                'probability': float(probability),
                'confidence': float(confidence)
            }
        except Exception as e:
            logger.info(f"Error making prediction:{e}")
            raise

    def generate_customer_id(self):
        """Generate unique customer id"""
        timestamp = int(time.time())
        unique_id = uuid4().hex[:5]
        return f"CUST_{timestamp}_{unique_id}"
    
    def get_message(self,prediction,confidence):
        """Generate human-redable message"""
        if prediction == 0:
            risk_level = "Low Risk" if confidence > 0.85 else "Medium Confidence"
            return f"Good credit customer - {risk_level}"
        
        else:
            risk_level = "High risk" if confidence> 0.85 else "Medium Confidence"
            return f"Bad credit customer - {risk_level}"


    def run_inference(self,input_data_dict):
        """Complete inference pipeline"""
        logger.info("-"*6)
        logger.info("Inference Pipeline start")
        logger.info("-"*6)

        start_time = time.time()

        try:
            # step 1 Generate customer id
            customer_id = self.generate_customer_id()
            logger.info(f"Generated customer id: {customer_id}")

            #step 2: prepare input
            X = self.prepare_input(input_data_dict)
            input_data_raw = X.copy()

            #step 3: feature engineering(before preprocessing)
            X_engineered = self.engineer_features(X)
            engineered_features = X_engineered[['debt_to_income', 'credit_utilization', 
                                                'income_per_age', 'employment_stability']].iloc[0].to_dict()
            
            #step 4: preprocess
            X_processed = self.preprocess(X_engineered)

            #step 5: Make prediction
            pred_result = self.predict(X_processed)

            #step 6: Generate message
            message = self.get_message(pred_result['prediction'],pred_result['confidence'])

            # calculate timing
            total_time = time.time() - start_time
            logger.info("-"*6)
            logger.info("INFERENCE PIPELINE COMPLETE")
            logger.info("-"*6)
            logger.info(f"Total time: {total_time*1000:.2f}ms")

            # Return complete result
            return {
                'customer_id': customer_id,
                'predicted_class': pred_result['prediction'],
                'probability': pred_result['probability'],
                'confidence': pred_result['confidence'],
                'model_name': self.model_name,
                'message': message,
                'inference_time_ms': total_time * 1000,
                # For database logging
                'input_data': input_data_raw.iloc[0].to_dict(),
                'engineered_features': engineered_features
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    from src.api.models_loader import ModelLoader

    logger = setup_logger('inference_service')

    # Load models
    loader = ModelLoader()
    loader.load_all(best_model_name='xgboost')

    # create inference service
    inference = InferenceService(
        loader,
        loader.get_preprocessor(),
        loader.get_feature_engineer()
    )

    # Test inference
    test_input = {
        'age': 35,
        'income': 75000,
        'debt': 15000,
        'credit_limit': 50000,
        'credit_used': 20000,
        'employment_years': 8,
        'employment_type': 'Salaried'

    }

    result = inference.run_inference(test_input)

    logger.info("Test Inference Result")
    for key, value in result.items():
        if key not in ['input_data','engineered_features']:
            logger.info(f"{key}:{value}")
            


