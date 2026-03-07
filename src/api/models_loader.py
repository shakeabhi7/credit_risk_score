import joblib
import logging
from src.config.config_loader import load_config
import yaml

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load trained Models and preprocessor"""

    def __init__(self,config_path="src/config/data_config.yaml"):

        self.config = load_config(config_path)

        self.models = {}
        self.preprocessor = None
        self.feature_engineer = None
        self.best_model = None
        self.best_model_name = None
        self.is_loaded = False

        # path from yaml
        self.artifacts_path = self.config["artifacts_path"]
        self.processed_path = self.config['processed_data_path']


    def load_models(self):
        """Load trained model"""

        logger.info("Loading trained models")
        try:
            self.models['random_forest'] = joblib.load(f"{self.artifacts_path}/random_forest_model.joblib")
            self.models["xgboost"] = joblib.load(
                f"{self.artifacts_path}/xgboost_model.joblib"
            )
            self.models["neural_network"] = joblib.load(
                f"{self.artifacts_path}/neural_network_model.joblib"
            )
            logger.info("All models loaded")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def load_preprocessor(self):
        """Load preprocessor"""

        logger.info("Loading preprocessor")

        try:
            self.preprocessor = joblib.load(f"{self.processed_path}/preprocessor.joblib")
            logger.info("Preprocessor loaded")
        
        except Exception as e:
            logger.info(f"Error loading preprocessor: {e}")
            raise
    
    def load_feature_engineer(self):
        """Load feature engineer"""

        logger.info("Loading feature engineer")

        try:
            self.feature_engineer = joblib.load(f"{self.processed_path}/feature_engineer.joblib")
            logger.info("Feature engineer loaded")

        except Exception as e:
            logger.info(f"Error loading feature engineer: {e}")
            raise

    def set_best_model(self,model_name = "xgboost"):
        """Set best model for predictions"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        self.best_model = self.models[model_name]
        self.best_model_name = model_name

        logger.info(f"Best model set to: {model_name}")

    def load_all(self,best_model_name="xgboost"):
        """Loadd all artifacts"""
        logger.info("-"*6)
        logger.info("Loading All Artifacts")
        logger.info("-"*6)

        try:
            self.load_models()
            self.load_preprocessor()
            self.load_feature_engineer()
            self.set_best_model(best_model_name)

            self.is_loaded=True

            logger.info("All Artifacts loaded successfully")
            logger.info(f"Best model:{self.best_model_name}")

            return True
        
        except Exception as e:
            logger.info(f"Error loading artifacts: {e}")
            return False
        
    def get_best_model(self):
        if not self.best_model:
            raise ValueError("Best model not set. Call load_all() first.")

        return self.best_model
    
    def get_preprocessor(self):

        if not self.preprocessor:
            raise ValueError("Preprocessor not loaded. Call load_all() first.")

        return self.preprocessor
    
    def get_feature_engineer(self):

        if not self.feature_engineer:
            raise ValueError("Feature engineer not loaded. Call load_all() first.")

        return self.feature_engineer
    

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger

    logger = setup_logger("models_loader")

    loader = ModelLoader()

    success = loader.load_all(best_model_name='xgboost')

    if success:
        logger.info("Ready for inference")

    else:
        logger.error("Failed to load the artifacts")
