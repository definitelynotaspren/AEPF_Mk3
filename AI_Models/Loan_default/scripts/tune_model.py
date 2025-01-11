"""Model tuning module for loan default prediction."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, f1_score, precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.metrics import make_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoanDefaultModelTuner:
    def __init__(self, data_path: str, output_dir: str):
        """Initialize model tuner."""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model parameters
        self.best_threshold = 0.5
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_split_data(self):
        """Load and split data with emphasis on balanced classes."""
        logger.info("Loading and preparing data...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop(['loan_status', 'id'], axis=1)
        y = df['loan_status']
        
        # Use stratified split to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(self.X_train)}")
        logger.info(f"Test set size: {len(self.X_test)}")
        
        # Calculate class weights
        self.class_weights = dict(zip(
            np.unique(y),
            len(y) / (len(np.unique(y)) * np.bincount(y))
        ))
        
        logger.info(f"Class weights: {self.class_weights}")
    
    def find_optimal_threshold(self, probabilities, actual):
        """Find optimal probability threshold for classification."""
        precisions, recalls, thresholds = precision_recall_curve(actual, probabilities)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]
    
    def tune_hyperparameters(self):
        """Perform grid search for optimal hyperparameters."""
        logger.info("Starting hyperparameter tuning...")
        
        # Adjust parameter grid to better handle class imbalance
        param_grid = {
            'max_depth': [4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'scale_pos_weight': [2, 3, 4],
            'min_child_weight': [3, 5, 7],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'n_estimators': [200, 300],
            'tree_method': ['hist'],
            'max_bin': [256],
            'grow_policy': ['lossguide'],
            'gamma': [0.1, 0.2],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [1, 2]
        }
        
        model = xgb.XGBClassifier(
            random_state=42,
            enable_categorical=True,
            eval_metric=['auc', 'logloss'],
            early_stopping_rounds=20,
            max_delta_step=1
        )
        
        # Custom scoring functions with proper handling of edge cases
        def custom_f1(y_true, y_pred):
            return f1_score(y_true, y_pred, zero_division=0)
        
        def custom_precision(y_true, y_pred):
            return precision_score(y_true, y_pred, zero_division=0)
        
        scoring = {
            'f1': make_scorer(custom_f1),
            'precision': make_scorer(custom_precision),
            'recall': 'recall',
            'roc_auc': 'roc_auc'
        }
        
        # Create stratified k-fold with shuffling
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring=scoring,
            refit='f1',
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True  # Get training scores for comparison
        )
        
        # Calculate balanced sample weights
        sample_weights = np.where(
            self.y_train == 1,
            1 / (self.y_train == 1).sum() * len(self.y_train) / 2,
            1 / (self.y_train == 0).sum() * len(self.y_train) / 2
        )
        
        # Create evaluation set for early stopping
        eval_set = [(self.X_test, self.y_test)]
        
        # Fit model with sample weights and evaluation set
        grid_search.fit(
            self.X_train,
            self.y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False  # Suppress XGBoost training output
        )
        
        # Log detailed results
        logger.info("\nCross-validation results:")
        for metric in scoring.keys():
            train_scores = grid_search.cv_results_[f'mean_train_{metric}']
            test_scores = grid_search.cv_results_[f'mean_test_{metric}']
            std_scores = grid_search.cv_results_[f'std_test_{metric}']
            
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"Train: {train_scores.mean():.3f}")
            logger.info(f"Test:  {test_scores.mean():.3f} (+/- {std_scores.mean():.3f})")
        
        logger.info(f"\nBest parameters: {grid_search.best_params_}")
        self.best_model = grid_search.best_estimator_
    
    def find_optimal_threshold_and_evaluate(self):
        """Find optimal threshold and evaluate model performance."""
        # Get probabilities
        train_probs = self.best_model.predict_proba(self.X_train)[:, 1]
        test_probs = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Find optimal threshold
        self.best_threshold = self.find_optimal_threshold(train_probs, self.y_train)
        logger.info(f"Optimal threshold: {self.best_threshold:.3f}")
        
        # Evaluate with optimal threshold
        test_preds = (test_probs >= self.best_threshold).astype(int)
        f1 = f1_score(self.y_test, test_preds)
        
        logger.info(f"F1 Score with optimal threshold: {f1:.3f}")
        
        # Plot probability distributions
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs[self.y_test == 0], bins=50, alpha=0.5, label='Non-default')
        plt.hist(test_probs[self.y_test == 1], bins=50, alpha=0.5, label='Default')
        plt.axvline(self.best_threshold, color='r', linestyle='--', label='Optimal Threshold')
        plt.title('Probability Distribution and Optimal Threshold')
        plt.xlabel('Predicted Probability of Default')
        plt.ylabel('Count')
        plt.legend()
        
        plot_path = self.output_dir / 'threshold_analysis.png'
        plt.savefig(plot_path)
        plt.close()
    
    def save_tuned_model(self):
        """Save the tuned model and configuration."""
        from joblib import dump
        
        # Save model
        model_path = self.output_dir / 'tuned_model.joblib'
        dump(self.best_model, model_path)
        
        # Save configuration
        config = {
            'optimal_threshold': self.best_threshold,
            'best_parameters': self.best_model.get_params(),
            'class_weights': self.class_weights
        }
        
        config_df = pd.DataFrame([config])
        config_df.to_json(self.output_dir / 'model_config.json')
        
        logger.info(f"Tuned model and configuration saved to {self.output_dir}")
    
    def tune_model(self):
        """Execute complete model tuning pipeline."""
        try:
            self.load_and_split_data()
            self.tune_hyperparameters()
            self.find_optimal_threshold_and_evaluate()
            self.save_tuned_model()
            
            logger.info("Model tuning completed successfully")
            
        except Exception as e:
            logger.error(f"Model tuning failed: {e}")
            raise

def main():
    """Main execution function."""
    try:
        base_path = Path(__file__).parent.parent
        data_path = base_path / 'data' / 'processed' / 'loan_data_preprocessed.csv'
        output_dir = base_path / 'models' / 'tuned'
        
        tuner = LoanDefaultModelTuner(str(data_path), str(output_dir))
        tuner.tune_model()
        
    except Exception as e:
        logger.error(f"Tuning process failed: {e}")
        raise

if __name__ == "__main__":
    main() 