import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import logging
import yaml
import os
import json

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self,
                 params_path="src/config/params.yaml",
                 data_config_path="src/config/data_config.yaml"):

        self.params = self._load_yaml(params_path)
        self.data_config = self._load_yaml(data_config_path)

        self.tuning_config = self.params["hyperparameter_tuning"]

        self.cv = self.tuning_config["cv"]
        self.n_jobs = self.tuning_config["n_jobs"]
        self.random_state = self.tuning_config['random_state']

        self.artifacts_path = self.data_config["artifacts_path"]
        self.processed_path = self.data_config['processed_data_path']
        self.files = self.data_config["files"]

        self.output_file = self.params["output"]["best_params_file"]

        self.best_params = {}
        self.grid_results = {}

    def _load_yaml(self, path):

        with open(path, "r") as f:
            return yaml.safe_load(f)
        
    # RANDOM FOREST
    def tune_random_forest(self,X_train,y_train):
        """Tune Random Forest model"""
        logger.info("-"*6)
        logger.info("Tuning: Random Forest HyperParameters")
        logger.info("-"*6)

        param_grid = self.tuning_config['random_forest']['param_grid']

        logger.info(f"Grid size: {self._calculate_grid_size(param_grid)}")

        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced"
        )

        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=self.cv,
            scoring="roc_auc",
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_params['random_forest'] = grid_search.best_params_
        self.grid_results['random_forest'] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }

        logger.info(f" Best RF Params Found!")
        logger.info(f" Best AUC-ROC Score: {grid_search.best_score_:.4f}")
        logger.info(f" Best Parameters:")
        for param, value in grid_search.best_params_.items():
            logger.info(f"     - {param}: {value}")
        
        return grid_search
    
    # XGBOOST
    def tune_xgboost(self, X_train, y_train):
        """Tune XGBoost model"""
        logger.info("-"*6)
        logger.info("Tuning: XGBoost HyperParameters")
        logger.info("-"*6)

        param_grid = self.tuning_config["xgboost"]["param_grid"]
        logger.info(f"Grid size (using RandomizedSearchCV): 20 random combinations")

        n_iter = self.tuning_config["xgboost"]["n_iter"]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        xgb = XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            verbosity=0
        )
        random_search = RandomizedSearchCV(
            xgb,
            param_grid,
            n_iter=n_iter,
            cv=self.cv,
            scoring="roc_auc",
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=42
        )

        random_search.fit(X_train, y_train)

        self.best_params['xgboost'] = random_search.best_params_
        self.grid_results['xgboost'] = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        }
        logger.info(f" Best XGBoost Params Found!")
        logger.info(f"   Best AUC-ROC Score: {random_search.best_score_:.4f}")
        logger.info(f"   Best Parameters:")
        for param, value in random_search.best_params_.items():
            logger.info(f"     - {param}: {value}")
        
        return random_search

    # NEURAL NETWORK
    def tune_neural_network(self, X_train, y_train):
        """Tune Neural Network hyperparameters"""
        logger.info("-"*6)
        logger.info("TUNING: Neural Network Hyperparameters")
        logger.info("-"*6)

        nn_config = self.tuning_config["neural_network"]

        param_grid = nn_config["param_grid"]

        n_iter = nn_config["n_iter"]

        max_iter = nn_config["max_iter"]

        validation_fraction = nn_config["validation_fraction"]

        logger.info(f"Grid size (using RandomizedSearchCV): 15 random combinations")

        nn = MLPClassifier(
            random_state=self.random_state,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=validation_fraction
        )
        random_search = RandomizedSearchCV(
            nn,
            param_grid,
            n_iter=n_iter,
            cv=self.cv,
            scoring="roc_auc",
            n_jobs=1,
            verbose=1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        self.best_params['neural_network'] = random_search.best_params_

        self.grid_results['neural_network'] = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        }

        logger.info(f" Best NN Params Found!")
        logger.info(f"   Best AUC-ROC Score: {random_search.best_score_:.4f}")
        logger.info(f"   Best Parameters:")
        for param, value in random_search.best_params_.items():
            logger.info(f"     - {param}: {value}")
        
        return random_search

    # SAVE PARAMS
    def save_best_params(self):

        try:
            filepath = os.path.join(self.artifacts_path, self.output_file)

            params_to_save = {}

            for model_name, params in self.best_params.items():

                params_to_save[model_name] = {

                    k: (
                        str(v)
                        if not isinstance(v, (int, float, bool, str, type(None)))
                        else v
                    )

                    for k, v in params.items()

                }

            with open(filepath, "w") as f:

                json.dump(params_to_save, f, indent=2)

            logger.info(f"Best hyperparameters saved to {filepath}")

        except Exception as e:

            logger.error(f"Error saving hyperparams: {e}")

            raise

    def _calculate_grid_size(self, param_grid):
        """Calculate total grid size"""
        size = 1
        for param_values in param_grid.values():
            size *= len(param_values)
        return size  

if __name__ == "__main__":

    from src.monitoring.logger import setup_logger

    logger = setup_logger("hyperparameter_tuner")

    tuner = HyperparameterTuner()

    # load paths from data_config.yaml

    X_train_path = os.path.join(
        tuner.processed_path,
        tuner.files["X_train_balanced"]
    )

    y_train_path = os.path.join(
        tuner.processed_path,
        tuner.files["y_train_balanced"]
    )

    X_train = pd.read_csv(X_train_path)

    y_train = pd.read_csv(y_train_path).squeeze()

    tuner.tune_random_forest(X_train, y_train)

    tuner.tune_xgboost(X_train, y_train)

    tuner.tune_neural_network(X_train, y_train)

    tuner.save_best_params()

    logger.info("Hyperparameter tuning complete.")




