import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class MLflowTracker:
    """Track model experiments with MLflow - Local OR DagsHub"""
    def __init__(self,experiment_name = os.getenv('EXPERIMENT_NAME'),use_dagshub=False):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of MLflow experiment
            use_dagshub: If True, connect to DagsHub cloud
                        If False, use local MLflow (default)
        """
        self.experiment_name = experiment_name
        self.use_dagshub = use_dagshub

        load_dotenv()

        try:
            if use_dagshub:
                self._setup_dagshub()
            else:
                self._setup_local()

            # create or get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)

            if experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id=experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
            logger.info(f"MlFlow tracking configured for: {experiment_name}")
        except Exception as e:
            logger.warning(f"Mlflow tracking setup failed:{e}")
            logger.warning(f"Continuing without MLflow tracking")
    
    def _setup_local(self):
        """Setup local MLflow tracking"""
        tracking_uri = 'http://localhost:5000'
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Local MLflow tracking:{tracking_uri}")

    def _setup_dagshub(self):
        """Setup Dagshub Mlflow tracking"""
        try:
            repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
            repo_name = os.getenv('DAGSHUB_REPO_NAME')
            token = os.getenv('DAGSHUB_TOKEN')

            if not all([repo_owner,repo_name,token]):
                raise ValueError("Missing credentials. ")
            
            # Set tracking URI

            # Set tracking URI
            dagshub_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
            mlflow.set_tracking_uri(dagshub_uri)
            
            # Set credentials
            os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
            os.environ['MLFLOW_TRACKING_PASSWORD'] = token
            
            logger.info(f" DagsHub MLflow tracking configured")
            logger.info(f"   Dashboard: {dagshub_uri}")
            logger.info(f"   Owner: {repo_owner}")
            logger.info(f"   Repo: {repo_name}")
        except Exception as e:
            logger.error(f" DagsHub setup failed: {e}")
            raise

    def log_model_training(self,model_name,model,X_train,X_test,y_train,y_test,hyperparams=None,description=""):
        """Log model training to Mlflow"""

        logger.info(f"Logging {model_name} to MLflow")
        try:
            with mlflow.start_run(run_name=f"{model_name}_run"):

                #Log Hyperparameters

                if hyperparams:
                    for param_name, param_value in hyperparams.items():
                        mlflow.log_name(param_name,param_value)

                
                # Get predcitions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:,1]

                #calculate metrics

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc_roc = roc_auc_score(y_test, y_pred_proba)
                # Log metrics
                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_metric('precision', precision)
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('f1_score', f1)
                mlflow.log_metric('auc_roc', auc_roc)
                
                # Log training data info
                mlflow.log_param('training_samples', len(X_train))
                mlflow.log_param('test_samples', len(X_test))
                mlflow.log_param('features', X_train.shape[1])

                #Log model

                if 'random_forest' in model_name.lower():
                    mlflow.sklearn.log_model(model,f"models/{model_name}")
                elif 'xgboost' in model_name.lower():
                    mlflow.xgboost.log_model(model, f"models/{model_name}")
                else:
                    mlflow.sklearn.log_model(model, f"models/{model_name}")

                # Log description and tags
                mlflow.set_tag('description',description)
                mlflow.set_tag('model_name',model_name)

                logger.info(f"{model_name} logged to Mlflow" )
                logger.info(f"   Accuracy: {accuracy:.4f}")
                logger.info(f"   AUC-ROC: {auc_roc:.4f}")
                logger.info(f"   F1-Score: {f1:.4f}")
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc
                } 
            
        except Exception as e:
            logger.info(f"Error logging to Mlflow: {e}")
            return None
        
    def log_best_model(self, model_name, best_score, model_type='classification'):
        """Log best model metadata"""

        try:
            with mlflow.start_run(run_name=f"{model_name}_best"):
                mlflow.log_metric('best_auc_roc',best_score)
                mlflow.set_tag('best_model',model_name)
                mlflow.set_tag('model_type',model_type)

            logger.info(f"Best model tagged:{model_name}")

        except Exception as e:
            logger.info(f"Could not log the best model: {e}")

    
    def compare_models(self,results_dict):
        """Log model comparison"""

        logger.info("-"*6)
        logger.info("MLFLOW: Model Comparison")
        logger.info("-"*6)
        try:
            with mlflow.start_run(run_name="model_comparison"):
                for model_name, metrics in results_dict.items():
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
                
                logger.info(" Model comparison logged to MLflow")
        except Exception as e:
            logger.warning(f" Could not log comparison: {e}")

    def end_tracking(self):
        """End MLflow tracking"""

        try:
            mlflow.end_run()
            logger.info("MLflow tracking ended")
        except Exception as e:
            logger.warning(f"Could not end MLflow run:{e}")


if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    
    logger = setup_logger('mlflow_tracker')
    
    # Example: Use local MLflow
    logger.info(" Testing LOCAL MLflow...")
    tracker_local = MLflowTracker(use_dagshub=False)
    logger.info("   View at: http://localhost:5000")
    
    # Example: Use DagsHub
    logger.info(" Testing DAGSHUB MLflow...")
    try:
        tracker_dagshub = MLflowTracker(use_dagshub=True)
        logger.info("DagsHub MLflow ready!")
    except Exception as e:
        logger.warning(f" DagsHub not configured: {e}")
        logger.warning("  To use DagsHub, set up .env file with credentials")




                   