"""
Machine learning models for microplastic hotspot prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicroplasticClassifier:
    """
    Ensemble classifier for microplastic hotspot prediction with 
    feature importance analysis and model interpretability
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.is_trained = False
        
    def _get_default_config(self) -> Dict:
        """Get default model configuration"""
        return {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'model': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'feature_selection': {
                'method': 'importance',
                'top_k': 50
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'is_hotspot') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training by handling missing values and feature selection
        
        Args:
            df: Input DataFrame with features
            target_col: Name of target column
            
        Returns:
            X, y arrays and feature names
        """
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Remove non-numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        X = df[numeric_cols].copy()
        y = df[target_col].values
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X.values, y, X.columns.tolist()
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Train Random Forest classifier with cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dictionary of model performance metrics
        """
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # Initialize Random Forest model
        self.model = RandomForestClassifier(**self.config['model'], 
                                          random_state=self.config['random_state'])
        
        # Train model
        logger.info("Training Random Forest classifier...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Model performance - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
        
        # Feature importance
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                   cv=self.config['cv_folds'], scoring='roc_auc')
        
        # Store metrics
        self.model_metrics = {
            'accuracy': accuracy,
            'auc': auc_score,
            'cross_validation': {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
        }
        
        self.is_trained = True
        self.feature_names = feature_names
        
        logger.info("Model training completed successfully")
        return self.model_metrics
    
    def predict(self, X: np.ndarray, return_probabilities: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions using trained Random Forest model
        
        Args:
            X: Feature matrix
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        predictions = self.model.predict(X)
        result = {'predictions': predictions}
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X)[:, 1]
            result['probabilities'] = probabilities
        
        return result
    
    def get_feature_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get feature importance rankings
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            List of top features by importance
        """
        if not self.feature_importance:
            raise ValueError("Feature importance not available. Train model first.")
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]
    
    def analyze_feature_groups(self, feature_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Analyze importance by feature groups
        
        Args:
            feature_groups: Dictionary mapping group names to feature lists
            
        Returns:
            Group importance analysis
        """
        group_importance = {}
        
        for group_name, features in feature_groups.items():
            # Calculate mean importance for features in this group
            group_features = [f for f in features if f in self.feature_importance]
            if group_features:
                mean_importance = np.mean([self.feature_importance[f] for f in group_features])
                group_importance[group_name] = mean_importance
            else:
                group_importance[group_name] = 0.0
        
        return group_importance
    
    def save_models(self, model_dir: str = "models/") -> None:
        """Save trained model and metadata"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, model_path / "random_forest_model.pkl")
        
        # Save metadata
        metadata = {
            'config': self.config,
            'metrics': self.model_metrics,
            'feature_importance': self.feature_importance,
            'feature_names': getattr(self, 'feature_names', []),
            'is_trained': self.is_trained
        }
        
        with open(model_path / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_models(self, model_dir: str = "models/") -> None:
        """Load saved model and metadata"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load metadata
        with open(model_path / "model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.config = metadata['config']
        self.model_metrics = metadata['metrics']
        self.feature_importance = metadata['feature_importance']
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        # Load the model
        model_file = model_path / "random_forest_model.pkl"
        if model_file.exists():
            self.model = joblib.load(model_file)
        else:
            raise FileNotFoundError("Random Forest model file not found")
        
        logger.info(f"Model loaded from {model_path}")


class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def generate_evaluation_report(classifier: MicroplasticClassifier, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        
        predictions = classifier.predict(X_test, return_probabilities=True)
        
        y_pred = predictions['predictions']
        y_proba = predictions.get('probabilities', None)
        
        report = {
            'model_performance': {
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
        }
        
        # ROC curve data
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            report['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
        
        return report


def main():
    """Main function for model training"""
    try:
        # Load feature-engineered data
        data_path = Path("data/features_marine_microplastics.csv")
        if not data_path.exists():
            raise FileNotFoundError("Feature-engineered data not found. Run feature engineering first.")
        
        df = pd.read_csv(data_path, parse_dates=['Date'])
        logger.info(f"Loaded {len(df)} records for model training")
        
        # Initialize classifier
        classifier = MicroplasticClassifier()
        
        # Prepare data
        X, y, feature_names = classifier.prepare_data(df)
        
        # Train models
        metrics = classifier.train(X, y, feature_names)
        
        # Get feature importance
        top_features = classifier.get_feature_importance(top_k=15)
        
        # Save models
        classifier.save_models()
        
        # Print results
        print("\n=== Model Training Results ===")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Random Forest Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AUC: {metrics['auc']:.3f}")
        print(f"  Cross-validation: {metrics['cross_validation']['mean']:.3f} ± {metrics['cross_validation']['std']:.3f}")
        
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(top_features[:10]):
            print(f"{i+1:2d}. {feature:<30} {importance:.4f}")
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()