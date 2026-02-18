import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import yaml
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train multiple models"""
    def __init__(self, 
             model_config_path="src/config/model_config.yaml",
             data_config_path="src/config/data_config.yaml"):

        self.model_config = self._load_yaml(model_config_path)
        self.data_config = self._load_yaml(data_config_path)

        self.processed_path = self.data_config["processed_data_path"]
        self.files = self.data_config["files"]

        self.rf_config = self.model_config["random_forest"]
        self.xgb_config = self.model_config["xgboost"]
        self.ann_config = self.model_config["ann"]
        self.random_state = self.model_config["data"]["random_state"]
        self.artifacts_path = self.data_config['artifacts_path']

        self.models = {}
        self.results = {}
 
    def _load_yaml(self,path):
        with open(path,"r") as file:
            return yaml.safe_load(file)
    
    # Random Forest

    def train_random_forest(self,X_train,y_train):
        """Train Random Forest model"""
        logger.info("-"*6)
        logger.info("Training Random Forest...")
        logger.info("-"*6)
        try:

            rf_model = RandomForestClassifier(
                n_estimators=self.rf_config["n_estimators"],
                max_depth = self.rf_config['max_depth'],
                min_samples_split = self.rf_config['min_samples_split'],
                random_state= self.rf_config['random_state'],
                n_jobs = self.rf_config['n_jobs'],
                class_weight="balanced"
            )

            rf_model.fit(X_train,y_train)
            logger.info(" Random Forest trained successfully!")

            self.models['random_forest'] = rf_model
            return rf_model
        except Exception as e:
            logger.error(f"Error training Random Forest:{e}")
            raise

    # XGBoost 
    def train_xgboost(self,X_train,y_train):
        """Train XGBoost model"""
        logger.info("-"*6)
        logger.info("Training XGBoost...")
        logger.info("-"*6)

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        try:
            xgb_model = XGBClassifier(
            n_estimators=self.xgb_config["n_estimators"],
            max_depth=self.xgb_config["max_depth"],
            learning_rate=self.xgb_config["learning_rate"],
            subsample=self.xgb_config["subsample"],
            colsample_bytree=self.xgb_config["colsample_bytree"],
            random_state=self.xgb_config["random_state"],
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            verbosity=0
            )
            
            xgb_model.fit(X_train,y_train)
            self.models["xgboost"] = xgb_model
            return xgb_model
            
        except Exception as e:
            logger.info(f"Error training XGBoost: {e}")
            raise
    
    # Neural Network

    def train_neural_network(self,X_train,y_train):
        """Train Neural Network model"""
        logger.info("-"*6)
        logger.info("Training Neural Network...")
        logger.info("-"*6)

        try:
            nn_model = MLPClassifier(
                hidden_layer_sizes=self.ann_config['layers'],
                activation = self.ann_config["activation"],
                max_iter = self.ann_config['epochs'],
                batch_size=self.ann_config['batch_size'],
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=self.ann_config['validation_split']

            )
            nn_model.fit(X_train,y_train)
            logger.info("Neural Network trained Successfully")

            self.models['neural_network'] = nn_model
            return nn_model
        except Exception as e:
            logger.info(f"Error training Neural Network: {e}")
            raise

        # Evaluation
    def evaluate_model(self,model_name,model,X_test,y_test):
        """Evaluate model performance"""
        logger.info(f"\nEvaluating {model_name}...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        self.results[model_name] = results

        logger.info(f"\n{model_name} Results:")
        for k, v in results.items():
            if isinstance(v, float):
                logger.info(f"{k}: {v:.4f}")

        return results
    
    # Save models
    def save_models(self):
        """Save all trained models"""
        logger.info("-"*6)
        logger.info("Saving models...")
        logger.info("-"*6)
        model_dir = self.artifacts_path
        os.makedirs(model_dir,exist_ok=True)
        for model_name, model in self.models.items():
            filepath = os.path.join(model_dir,f"{model_name}.joblib")
            joblib.dump(model,filepath)
            logger.info(f"Saved: {filepath}")
        
        # save results
        results_path = os.path.join(model_dir,"model_results.joblib")
        joblib.dump(self.results,results_path)
        logger.info(f"Saved;{results_path}")

    #Best model

    def get_best_model(self):
        """Get model with best AUC-ROC"""
        best_model_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['auc_roc'])
        best_auc = self.results[best_model_name]['auc_roc']
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"AUC-ROC : {best_auc:.4f}")

        return best_model_name,self.models[best_model_name]
    
if __name__ == "__main__":

    from src.monitoring.logger import setup_logger

    logger = setup_logger("model_trainer")

    trainer = ModelTrainer()

    # Config-driven paths
    X_train_path = os.path.join(trainer.processed_path, trainer.files["X_train_balanced"])
    y_train_path = os.path.join(trainer.processed_path, trainer.files["y_train_balanced"])
    X_test_path = os.path.join(trainer.processed_path, trainer.files["X_test"])
    y_test_path = os.path.join(trainer.processed_path, trainer.files["y_test"])

    # Load data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    # Train
    rf = trainer.train_random_forest(X_train, y_train)
    xgb = trainer.train_xgboost(X_train, y_train)
    nn = trainer.train_neural_network(X_train, y_train)

    # Evaluate
    trainer.evaluate_model("random_forest", rf, X_test, y_test)
    trainer.evaluate_model("xgboost", xgb, X_test, y_test)
    trainer.evaluate_model("neural_network", nn, X_test, y_test)

    trainer.save_models()
    trainer.get_best_model()




        



