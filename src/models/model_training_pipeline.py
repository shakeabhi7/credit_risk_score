import os
import time
import yaml
from src.monitoring.logger import setup_logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import sys
from src.preprocessing.imbalance_handler import ImbalanceHandler
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.models.mlflow_tracker import MLflowTracker
from src.database.prediction_logger import PredictionLogger

logger = setup_logger('model_training_pipeline')

# Load YAML CONFIG

def load_configs():

    with open("src/config/data_config.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    with open("src/config/params.yaml", "r") as f:
        params_config = yaml.safe_load(f)

    return data_config, params_config

# STEP 1: LOAD DATA

def load_data(data_config):
    """Load training and test data"""
    logger.info( "-"*6)
    logger.info("STEP 1: LOADING DATA")
    logger.info("-"*6)

    processed_path = data_config["processed_data_path"]
    files = data_config["files"]
    artifacts_path = data_config['artifacts_path']

    X_train = pd.read_csv(os.path.join(processed_path, files["X_train"]))
    X_test = pd.read_csv(os.path.join(processed_path, files["X_test"]))
    y_train = pd.read_csv(os.path.join(processed_path, files["y_train"])).squeeze()
    y_test = pd.read_csv(os.path.join(processed_path, files["y_test"])).squeeze()

    logger.info(f" Training data: {X_train.shape}")
    logger.info(f" Test data: {X_test.shape}")
    logger.info(f" Training target distribution:")
    logger.info(f"   Good (0): {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
    logger.info(f"   Bad (1): {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# STEP 2: HANDLE IMBALANCE

def handle_imbalance(X_train, y_train, data_config):
    """Handle class imbalance using SMOTE"""
    logger.info( "-"*6)
    logger.info("STEP 2: HANDLING CLASS IMBALANCE (SMOTE)")
    logger.info("-"*6)
    
    handler = ImbalanceHandler()

    X_bal, y_bal = handler.fit_resample(X_train, y_train)

    processed_path = data_config["processed_data_path"]
    files = data_config["files"]

    X_bal.to_csv(os.path.join(processed_path, files["X_train_balanced"]), index=False)
    y_bal.to_csv(os.path.join(processed_path, files["y_train_balanced"]), index=False)
    logger.info(" Balanced data saved")

    return X_bal, y_bal

# STEP 3: HYPERPARAMETER TUNING
def tune_hyperparameters(X_bal, y_bal, params_config, data_config):
    """Tune hyperparameters for all models"""
    logger.info( "-"*6)
    logger.info("STEP 3: HYPERPARAMETER TUNING")
    logger.info("-"*6)

    tuning_config = params_config["hyperparameter_tuning"]

    tuner = HyperparameterTuner()

    logger.info(" Tuning Random Forest...")
    tuner.tune_random_forest(X_bal, y_bal)

    logger.info(" Tuning XGBoost...")
    tuner.tune_xgboost(X_bal, y_bal)

    logger.info(" Tuning Neural Network...")
    tuner.tune_neural_network(X_bal, y_bal)

    # Save best params
    tuner.save_best_params()

    return tuner

# STEP 4: TRAIN MODELS WITH BEST PARAMS

def train_models(X_bal, y_bal, tuner, params_config):
    """Train models using best hyperparameters"""
    logger.info( "-"*6)
    logger.info("STEP 4: TRAINING MODELS WITH BEST PARAMS")
    logger.info("-"*6)

    models = {}

    random_state = params_config["hyperparameter_tuning"]["random_state"]

    # Random Forest
    logger.info(" Training Random Forest...")
    rf_model = RandomForestClassifier(
        **tuner.best_params["random_forest"],
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_bal, y_bal)
    models['random_forest'] = rf_model
    logger.info(" Random Forest trained")

     # XGBoost
    logger.info(" Training XGBoost...")
    xgb_model=XGBClassifier(
        **tuner.best_params["xgboost"],
        random_state=random_state,
        verbosity=0
    )
    xgb_model.fit(X_bal, y_bal)
    models['xgboost'] = xgb_model
    logger.info(" XGBoost trained")

    nn_config = params_config["hyperparameter_tuning"]["neural_network"]

    # Neural Network
    logger.info("Training Neural Network...")
    nn_model = MLPClassifier(
        **tuner.best_params["neural_network"],
        random_state=random_state,
        max_iter=nn_config["max_iter"],
        validation_fraction=nn_config["validation_fraction"],
        early_stopping=True
    )
    nn_model.fit(X_bal, y_bal)
    models['neural_network'] = nn_model
    logger.info(" Neural Network trained")
    
    return models

# STEP 5: EVALUATE MODELS
def evaluate_models(models, X_test, y_test):
    """Evaluate all models"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score, confusion_matrix)
    
    logger.info( "-"*6)
    logger.info("STEP 5: EVALUATING MODELS")
    logger.info("-"*6)
    
    results = {}
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob)
        }

    return results

# STEP 6: MLFLOW TRACKING
def log_to_mlflow(models, X_train_balanced, X_test, y_test, results, tuner, use_dagshub=False):
    """Log experiments to MLflow"""
    logger.info( "-"*6)
    logger.info("STEP 6: MLFLOW TRACKING")
    logger.info("-"*6)

    # Initialize MLflow tracker
    mlflow_tracker = MLflowTracker(use_dagshub=use_dagshub)


    for model_name, model in models.items():
        logger.info(f"Logging {model_name} to MLflow...")
        mlflow_tracker.log_model_training(
            model_name=model_name,
            model=model,
            X_train=X_train_balanced,
            X_test=X_test,
            y_test=y_test,
            hyperparams=tuner.best_params.get(model_name, {}),
            description=f"{model_name.upper()} with tuned hyperparameters",
        )

    # LOG BEST MODEL
    best_model_name = max(
        results,
        key=lambda name: results[name]["auc_roc"]
    )
    best_auc = results[best_model_name]["auc_roc"]
    logger.info(f"Best model: {best_model_name} (AUC={best_auc:.4f})")
    mlflow_tracker.log_best_model(
        model_name=best_model_name,
        auc_score=best_auc
    )
    #LOG COMPARISON
    mlflow_tracker.compare_models(results)
    logger.info("All models logged to MLflow successfully")
    return best_model_name

# STEP 7: DATABASE LOGGING
def log_to_database(results, X_bal, X_test):
    """Log training sessions to MongoDB"""
    logger.info( "-"*6)
    logger.info("STEP 7: DATABASE LOGGING")
    logger.info("-"*6)

    pred_logger = PredictionLogger()

    if not pred_logger.connected:
        logger.info("Not Connected Returned")
        return
    
    for model_name, result in results.items():

        session_data = {
            "metadata": {
                "timestamp": pd.Timestamp.now(),
                "model_version": "v1.0"
            },
            "input_data": {
                "training_samples": len(X_bal),
                "test_samples": len(X_test),
                "features_count": X_bal.shape[1]
            },
            "engineered_features": {
                "features_list": X_bal.columns.tolist()
            },
            'preprocessing_info': {
                    'scaler_type': 'StandardScaler',
                    'encoder_type': 'LabelEncoder'
                },
            "prediction": result | {"model_name": model_name}
        }
        session_id = pred_logger.log_training_session(session_data)
        if session_id:
                logger.info(f" {model_name} training logged to DB")
        
    pred_logger.close()


# STEP 8: SAVE MODELS
def save_models(models, results, data_config):
    """Save trained models"""
    logger.info( "-"*6)
    logger.info("STEP 8: SAVING MODELS")
    logger.info("-"*6)

    import joblib
    artifacts_path = data_config["artifacts_path"]
    for name, model in models.items():

        joblib.dump(
            model,
            os.path.join(artifacts_path, f"{name}_model.joblib")
        )
        logger.info(f" Saved: {artifacts_path}")

    joblib.dump(
        results,
        os.path.join(artifacts_path, "model_results.joblib")
    )
    logger.info(f" Saved: {artifacts_path}")


# STEP 9: CREATE VISUALIZATIONS
def create_visualizations(models, X_test, y_test, results):
    """Create evaluation visualizations"""
    logger.info( "-"*6)
    logger.info("STEP 9: CREATING VISUALIZATIONS")
    logger.info("-"*6)
    
    from src.models.model_evaluator import ModelEvaluator
    
    evaluator = ModelEvaluator()
    evaluator.load_models()
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves(X_test, y_test)
    evaluator.plot_metrics_comparison()
    evaluator.plot_feature_importance()
    evaluator.print_summary()
    
    logger.info(" Visualizations created")


#MAIN PIPELINE
def main():
    """Run complete Day 4 pipeline"""
    
    logger.info("-"*10)
    logger.info("COMPLETE MODEL TRAINING PIPELINE")
    logger.info("-"*10)
    
    start = time.time()

    try:
        data_config, params_config = load_configs()

        # Step 1: Load data
        X_train, X_test, y_train, y_test = load_data(data_config)

        # Step 2: Handle class imbalance
        X_bal, y_bal = handle_imbalance(X_train, y_train, data_config)

        # Step 3: Hyperparameter tuning
        tuner = tune_hyperparameters(X_bal, y_bal, params_config, data_config)

        # Step 4: Train models
        models = train_models(X_bal, y_bal, tuner, params_config)

        # Step 5: Evaluate models
        results = evaluate_models(models, X_test, y_test)

        # Step 6: MLflow tracking
        best_model_name = log_to_mlflow(models, X_bal, X_test, y_test, results, tuner, use_dagshub=True)

        # Step 7: Database logging
        log_to_database(results, X_bal, X_test)

        # Step 8: Save models       
        save_models(models, results, data_config)

        # Step 9: Visualizations
        create_visualizations(models, X_test, y_test, results)


        end = time.time()

        logger.info(f"Pipeline completed in {(end-start)/60:.2f} minutes")
        logger.info("-"*6)
        logger.info("DAY 4 PIPELINE - COMPLETE!")
        logger.info("-"*6)

        return True

    
    except Exception as e:
        logger.info("Pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
