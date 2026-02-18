import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             roc_auc_score, classification_report)
import joblib
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, model_dir='data/processed'):
        self.model_dir = model_dir
        self.models = {}
        self.results = {}
    
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading models...")
        
        for model_name in ['random_forest', 'xgboost', 'neural_network']:
            filepath = os.path.join(self.model_dir, f'{model_name}_model.joblib')
            self.models[model_name] = joblib.load(filepath)
            logger.info(f" Loaded: {model_name}")
        
        results_filepath = os.path.join(self.model_dir, 'model_results.joblib')
        self.results = joblib.load(results_filepath)
        logger.info(f"Loaded results")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        logger.info("\nCreating confusion matrix plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, xticklabels=['Good', 'Bad'], 
                       yticklabels=['Good', 'Bad'])
            axes[idx].set_title(f'{model_name.upper()}\nAccuracy: {results["accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('notebooks/17_confusion_matrices.png', dpi=100, bbox_inches='tight')
        logger.info(" Saved: 17_confusion_matrices.png")
        plt.close()
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        logger.info("\nCreating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name.upper()} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('notebooks/18_roc_curves.png', dpi=100, bbox_inches='tight')
        logger.info("Saved: 18_roc_curves.png")
        plt.close()
    
    def plot_metrics_comparison(self):
        """Compare metrics across models"""
        logger.info("\nCreating metrics comparison plots...")
        
        metrics_data = []
        for model_name, results in self.results.items():
            metrics_data.append({
                'Model': model_name.upper(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results['auc_roc']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Bar plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # All metrics comparison
        df_metrics.set_index('Model').plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Metrics Comparison')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim([0, 1])
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.3, axis='y')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # AUC-ROC focus
        df_metrics.set_index('Model')['AUC-ROC'].plot(kind='bar', ax=axes[1], color='steelblue')
        axes[1].set_title('AUC-ROC Score Comparison')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_ylim([0.5, 1])
        axes[1].grid(alpha=0.3, axis='y')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('notebooks/19_metrics_comparison.png', dpi=100, bbox_inches='tight')
        logger.info(" Saved: 19_metrics_comparison.png")
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        logger.info("\nCreating feature importance plots...")
        
        # Load feature names
        X_train = pd.read_csv('data/processed/X_train_balanced.csv')
        feature_names = X_train.columns.tolist()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Random Forest
        rf_model = self.models['random_forest']
        rf_importance = rf_model.feature_importances_
        rf_indices = np.argsort(rf_importance)[-10:]
        
        axes[0].barh(range(len(rf_indices)), rf_importance[rf_indices])
        axes[0].set_yticks(range(len(rf_indices)))
        axes[0].set_yticklabels([feature_names[i] for i in rf_indices])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Random Forest - Top 10 Features')
        axes[0].grid(alpha=0.3, axis='x')
        
        # XGBoost
        xgb_model = self.models['xgboost']
        xgb_importance = xgb_model.feature_importances_
        xgb_indices = np.argsort(xgb_importance)[-10:]
        
        axes[1].barh(range(len(xgb_indices)), xgb_importance[xgb_indices])
        axes[1].set_yticks(range(len(xgb_indices)))
        axes[1].set_yticklabels([feature_names[i] for i in xgb_indices])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('XGBoost - Top 10 Features')
        axes[1].grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('notebooks/20_feature_importance.png', dpi=100, bbox_inches='tight')
        logger.info(" Saved: 20_feature_importance.png")
        plt.close()
    
    def print_summary(self):
        """Print evaluation summary"""
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION SUMMARY")
        logger.info("="*60)
        
        for model_name, results in self.results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  Precision: {results['precision']:.4f}")
            logger.info(f"  Recall: {results['recall']:.4f}")
            logger.info(f"  F1-Score: {results['f1_score']:.4f}")
            logger.info(f"  AUC-ROC: {results['auc_roc']:.4f}")

if __name__ == "__main__":
    from src.monitoring.logger import setup_logger
    
    logger = setup_logger('model_evaluator')
    
    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    # Evaluate
    evaluator = ModelEvaluator()
    evaluator.load_models()
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves(X_test, y_test)
    evaluator.plot_metrics_comparison()
    evaluator.plot_feature_importance()
    evaluator.print_summary()
    
    logger.info("\n Evaluation complete!")