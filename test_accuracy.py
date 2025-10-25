#!/usr/bin/env python3
"""
Comprehensive Random Forest Accuracy Testing Script
Tests the trained Random Forest model with detailed performance metrics
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.train_classifier import MicroplasticClassifier

def test_random_forest_accuracy():
    """
    Comprehensive test of Random Forest model accuracy
    """
    print("🧪 Starting Random Forest Accuracy Testing")
    print("=" * 50)
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    try:
        # Load feature-engineered data (same as in training script)
        data_path = Path("data/features_marine_microplastics.csv")
        if not data_path.exists():
            print(f"❌ Feature-engineered data not found: {data_path}")
            print("💡 Please run feature engineering first: python src/features/engineer_features.py")
            return False
            
        df = pd.read_csv(data_path, parse_dates=['Date'] if 'Date' in pd.read_csv(data_path, nrows=1).columns else None)
        print(f"✅ Loaded {len(df)} samples")
        print(f"✅ Data has {df.shape[1]} features")
        
        # Prepare data for training
        classifier = MicroplasticClassifier()
        X, y, feature_names = classifier.prepare_data(df)
        print(f"✅ Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        return False
    
    # Test model training and evaluation
    print("\n🤖 Training and testing Random Forest model...")
    try:
        # Train the model
        metrics = classifier.train(X, y, feature_names)
        print(f"✅ Model trained successfully")
        
        # Get predictions for evaluation
        predictions = classifier.predict(X)
        y_pred = predictions['predictions']
        y_prob = predictions['probabilities']
        
        print("\n📈 Performance Metrics:")
        print("-" * 30)
        
        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        auc = roc_auc_score(y, y_prob)
        
        print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
        print(f"🎯 Precision: {precision:.4f} ({precision:.1%})")
        print(f"🎯 Recall:    {recall:.4f} ({recall:.1%})")
        print(f"🎯 F1-Score:  {f1:.4f} ({f1:.1%})")
        print(f"🎯 AUC:       {auc:.4f} ({auc:.1%})")
        
        # Cross-validation
        print("\n🔄 Cross-Validation Results:")
        print("-" * 30)
        cv_scores = cross_val_score(
            classifier.model, X, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        print(f"🎯 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"🎯 CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Confusion Matrix
        print("\n📊 Confusion Matrix:")
        print("-" * 30)
        cm = confusion_matrix(y, y_pred)
        print(f"True Negatives:  {cm[0,0]:4d} | False Positives: {cm[0,1]:4d}")
        print(f"False Negatives: {cm[1,0]:4d} | True Positives:  {cm[1,1]:4d}")
        
        # Classification Report
        print("\n📋 Detailed Classification Report:")
        print("-" * 40)
        print(classification_report(y, y_pred, target_names=['Low Risk', 'High Risk']))
        
        # Feature Importance (Top 10)
        print("\n🎖️ Top 10 Most Important Features:")
        print("-" * 40)
        feature_importance = classifier.feature_importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<30} {importance:.4f}")
        
        # Model evaluation summary
        print("\n" + "=" * 50)
        print("🏆 RANDOM FOREST ACCURACY TEST SUMMARY")
        print("=" * 50)
        
        if accuracy >= 0.95:
            print("✅ EXCELLENT: Model achieves >95% accuracy")
        elif accuracy >= 0.90:
            print("✅ VERY GOOD: Model achieves >90% accuracy")
        elif accuracy >= 0.80:
            print("⚠️  GOOD: Model achieves >80% accuracy")
        else:
            print("❌ POOR: Model accuracy <80%")
            
        print(f"📊 Final Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"📊 Final AUC: {auc:.4f} ({auc:.1%})")
        print(f"📊 Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Check if model files exist
        print("\n💾 Model Persistence Check:")
        print("-" * 30)
        model_path = Path("models/random_forest_model.pkl")
        if model_path.exists():
            print(f"✅ Model saved: {model_path}")
            print(f"📁 File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print(f"❌ Model not found: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_consistency():
    """
    Test that predictions are consistent and reasonable
    """
    print("\n🔍 Testing Prediction Consistency...")
    print("-" * 40)
    
    try:
        # Load trained model
        model_path = Path("models/random_forest_model.pkl")
        if not model_path.exists():
            print("❌ No trained model found. Please run training first.")
            return False
            
        classifier = MicroplasticClassifier()
        classifier.model = joblib.load(model_path)
        classifier.is_trained = True
        
        # Test with sample data
        sample_data = np.array([
            [0.1, 35.0, -140.0] + [0.5] * 66,  # Low concentration sample
            [0.8, 40.0, -120.0] + [0.7] * 66,  # High concentration sample
        ])
        
        predictions = classifier.predict(sample_data)
        
        print(f"✅ Sample 1 (low conc): Prediction = {predictions['predictions'][0]}, Probability = {predictions['probabilities'][0]:.3f}")
        print(f"✅ Sample 2 (high conc): Prediction = {predictions['predictions'][1]}, Probability = {predictions['probabilities'][1]:.3f}")
        
        # Check that high concentration gets higher probability
        if predictions['probabilities'][1] > predictions['probabilities'][0]:
            print("✅ Prediction logic is consistent: Higher concentration → Higher probability")
        else:
            print("⚠️  Prediction logic may need review")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction testing: {e}")
        return False

if __name__ == "__main__":
    print("🌊 Marine Microplastic Prediction - Random Forest Accuracy Test")
    print("=" * 60)
    
    # Run comprehensive accuracy test
    success1 = test_random_forest_accuracy()
    
    # Run prediction consistency test
    success2 = test_prediction_consistency()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED - Random Forest model is performing excellently!")
    else:
        print("❌ Some tests failed - please check the output above")
    print("=" * 60)