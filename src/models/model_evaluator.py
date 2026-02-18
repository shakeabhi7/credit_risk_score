import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             roc_auc_score, classification_report)
import joblib
import logging
import os
import numpy as np

artifact_path = 'data\artifacts'

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    def __init__(self, model_dir=None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, 'data', 'artifacts')

        self.model_dir = model_dir

        self.models = {}
        self.results = {}
    
     # Create visualization directory
        self.visual_dir = "src/models/evaluation_visuals"
        os.makedirs(self.visual_dir, exist_ok=True)
    
    #Load Models

    def load_models(self):
        """Load all trained models"""
        logger.info("Loading models...")


        for model_name in ['random_forest', 'xgboost', 'neural_network']:
            filepath = os.path.join(self.model_dir, f'{model_name}.joblib')
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                logger.info(f" Loaded: {model_name}")
            else:
                logger.warning(f" Model not found: {model_name}")

        results_filepath = os.path.join(self.model_dir, 'model_results.joblib')
        if os.path.exists(results_filepath):
            self.results = joblib.load(results_filepath)
            logger.info(" Loaded results")
        else:
            logger.warning(" model_results.joblib not found")

    # CONFUSION MATRICES

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        logger.info("Creating confusion matrix plots...")

        for model_name, results in self.results.items():
            cm = results['confusion_matrix']

            plt.figure(figsize=(6, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False,
                xticklabels=['Good', 'Bad'],
                yticklabels=['Good', 'Bad']
            )

            plt.title(f"{model_name.upper()} Confusion Matrix Accuracy: {results['accuracy']:.3f}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()

            save_path = os.path.join(self.visual_dir,
                                     f"{model_name}_confusion_matrix.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

            logger.info(f" Saved: {save_path}")
    
    # ROC CURVES

    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        logger.info("Creating ROC curves...")

        plt.figure(figsize=(8, 6))

        for model_name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                logger.warning(f"{model_name} does not support predict_proba")
                continue

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr,
                     linewidth=2,
                     label=f"{model_name.upper()} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.visual_dir,
                                 "roc_curves_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        logger.info(f" Saved: {save_path}")

    def plot_metrics_comparison(self):
        """Compare metrics across models"""
        logger.info("\nCreating metrics comparison plots...")

        metrics_data = []
        for model_name, results in self.results.items():
            metrics_data.append({
                "Model": model_name.upper(),
                "Accuracy": results["accuracy"],
                "Precision": results["precision"],
                "Recall": results["recall"],
                "F1-Score": results["f1_score"],
                "AUC-ROC": results["auc_roc"]
            })

        df_metrics = pd.DataFrame(metrics_data)

        df_melted = df_metrics.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        )

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=df_melted,
            x="Model",
            y="Score",
            hue="Metric"
        )

        plt.ylim(0, 1)
        plt.title("Model Metrics Comparison")
        plt.tight_layout()

        save_path = os.path.join(self.visual_dir,
                                 "metrics_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        logger.info(f" Saved: {save_path}")

    # FEATURE IMPORTANCE
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        logger.info("Creating feature importance plots...")

        train_path = "data/processed/X_train_balanced.csv"

        if not os.path.exists(train_path):
            logger.warning(" X_train_balanced.csv not found")
            return

        X_train = pd.read_csv(train_path)
        feature_names = X_train.columns.tolist()

        for model_key in ["random_forest", "xgboost"]:
            if model_key not in self.models:
                continue

            model = self.models[model_key]

            if not hasattr(model, "feature_importances_"):
                logger.warning(f"{model_key} has no feature_importances_")
                continue

            importance = model.feature_importances_

            indices = np.argsort(importance)[-10:]
            top_features = [feature_names[i] for i in indices]
            top_importance = importance[indices]

            plt.figure(figsize=(8, 6))

            sns.barplot(
                x=top_importance,
                y=top_features
            )

            plt.title(f"{model_key.upper()} - Top 10 Feature Importance")
            plt.xlabel("Importance")
            plt.tight_layout()

            save_path = os.path.join(self.visual_dir,
                                     f"{model_key}_feature_importance.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

            logger.info(f" Saved: {save_path}")

    def print_summary(self):
        """Print evaluation summary"""
        logger.info("-"*6)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("-"*6)

        for model_name, results in self.results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  Precision: {results['precision']:.4f}")
            logger.info(f"  Recall: {results['recall']:.4f}")
            logger.info(f"  F1-Score: {results['f1_score']:.4f}")
            logger.info(f"  AUC-ROC: {results['auc_roc']:.4f}")

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger

    logger = setup_logger("model_evaluator")

    # Load test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    evaluator = ModelEvaluator()
    evaluator.load_models()

    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves(X_test, y_test)
    evaluator.plot_metrics_comparison()
    evaluator.plot_feature_importance()
    evaluator.print_summary()

    logger.info("Evaluation complete!")