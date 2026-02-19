import os
import json
import time
import yaml
from src.monitoring.logger import setup_logger
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from src.preprocessing.imbalance_handler import ImbalanceHandler
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.models.mlflow_tracker import MLflowTracker
from src.database.prediction_logger import PredictionLogger

logger = setup_logger('model_training_pipeline')

# Step 1 Load Data
def load_data():
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
