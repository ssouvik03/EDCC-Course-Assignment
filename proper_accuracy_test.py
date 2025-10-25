#!/usr/bin/env python3
"""
PROPER Random Forest Accuracy Testing - NO DATA LEAKAGE!
This script properly respects train/test splits and doesn't test on training data
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.train_classifier import MicroplasticClassifier

def proper_accuracy_test():
    """
    PROPER test that respects train/test separation - NO CHEATING!
    """
    print("🚨 PROPER Random Forest Testing - NO DATA LEAKAGE!")
    print("=" * 60)
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    try:
        data_path = Path("data/features_marine_microplastics.csv")
        if not data_path.exists():
            print(f"❌ Feature-engineered data not found: {data_path}")
            return False
            
        df = pd.read_csv(data_path, parse_dates=['Date'] if 'Date' in pd.read_csv(data_path, nrows=1).columns else None)
        print(f"✅ Loaded {len(df)} samples")
        
        # Prepare data
        classifier = MicroplasticClassifier()
        X, y, feature_names = classifier.prepare_data(df)
        print(f"✅ Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        return False
    
    # PROPER train/test split - DO THIS OURSELVES!
    print("\n🔪 Creating proper train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"📊 Training hotspots: {y_train.sum()}/{len(y_train)} ({y_train.mean():.1%})")
    print(f"📊 Test hotspots: {y_test.sum()}/{len(y_test)} ({y_test.mean():.1%})")
    
    # Train model ONLY on training data
    print("\n🤖 Training model on TRAINING DATA ONLY...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Train ONLY on training data
        model.fit(X_train, y_train)
        print("✅ Model trained on training data only")
        
        # Test on UNSEEN test data
        print("\n🧪 Testing on UNSEEN test data...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate HONEST metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n📈 HONEST Performance Metrics (Test Set Only):")
        print("-" * 50)
        print(f"🎯 Test Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
        print(f"🎯 Test Precision: {precision:.4f} ({precision:.1%})")
        print(f"🎯 Test Recall:    {recall:.4f} ({recall:.1%})")
        print(f"🎯 Test F1-Score:  {f1:.4f} ({f1:.1%})")
        print(f"🎯 Test AUC:       {auc:.4f} ({auc:.1%})")
        
        # Cross-validation for even more honest assessment
        print("\n🔄 Cross-Validation (5-Fold):")
        print("-" * 30)
        cv_scores = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        print(f"🎯 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"🎯 CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Test set confusion matrix
        print("\n📊 Test Set Confusion Matrix:")
        print("-" * 30)
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives:  {cm[0,0]:4d} | False Positives: {cm[0,1]:4d}")
        print(f"False Negatives: {cm[1,0]:4d} | True Positives:  {cm[1,1]:4d}")
        
        # Detailed classification report for test set
        print("\n📋 Test Set Classification Report:")
        print("-" * 40)
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        # Check if we still have perfect separation on test set
        print("\n🔍 Analyzing Test Set Separation:")
        print("-" * 40)
        
        # Get concentration values for test set (assuming it's the first feature)
        test_concentrations = X_test[:, 0]  # Concentration should be first feature
        
        test_data_analysis = pd.DataFrame({
            'concentration': test_concentrations,
            'actual': y_test,
            'predicted': y_pred
        })
        
        # Check separation on test set
        non_hotspot_max = test_data_analysis[test_data_analysis['actual'] == 0]['concentration'].max()
        hotspot_min = test_data_analysis[test_data_analysis['actual'] == 1]['concentration'].min()
        
        print(f"Test set - Highest non-hotspot: {non_hotspot_max:.3f}")
        print(f"Test set - Lowest hotspot: {hotspot_min:.3f}")
        
        if non_hotspot_max < hotspot_min:
            print("🚨 Even test set has perfect separation!")
            print("   This confirms the data is synthetically generated with clear rules")
        else:
            print("✅ Test set has some overlap - more realistic!")
        
        # Error analysis
        errors = test_data_analysis[test_data_analysis['actual'] != test_data_analysis['predicted']]
        print(f"\n❌ Misclassified samples in test set: {len(errors)}")
        if len(errors) > 0:
            print("Error details:")
            print(errors[['concentration', 'actual', 'predicted']].head())
        
        return accuracy, cv_scores.mean()
        
    except Exception as e:
        print(f"❌ Error in model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_data_leakage():
    """
    Check for potential data leakage issues in the dataset
    """
    print("\n🕵️ ANALYZING POTENTIAL DATA LEAKAGE ISSUES")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv("data/features_marine_microplastics.csv")
    
    # Check if features contain future information
    suspicious_features = []
    
    # Check for features that might leak target information
    correlation_with_target = df.corr()['is_hotspot'].abs().sort_values(ascending=False)
    high_correlation_features = correlation_with_target[correlation_with_target > 0.8]
    
    print("🚨 Features with suspiciously high correlation to target:")
    print("-" * 50)
    for feature, corr in high_correlation_features.items():
        if feature != 'is_hotspot':
            print(f"{feature:<30} {corr:.4f}")
            if corr > 0.9:
                suspicious_features.append(feature)
    
    if len(suspicious_features) > 0:
        print(f"\n⚠️  Highly suspicious features (>90% correlation): {suspicious_features}")
        print("   These might be derived from the target variable!")
    
    return suspicious_features

if __name__ == "__main__":
    print("🌊 PROPER Marine Microplastic Accuracy Test - NO CHEATING!")
    print("=" * 65)
    
    # Check for data leakage first
    suspicious = analyze_data_leakage()
    
    # Run proper accuracy test
    result = proper_accuracy_test()
    
    print("\n" + "=" * 65)
    print("🏁 FINAL VERDICT:")
    print("=" * 65)
    
    if result:
        test_acc, cv_acc = result
        print(f"📊 Honest Test Accuracy: {test_acc:.4f} ({test_acc:.1%})")
        print(f"📊 Cross-Validation: {cv_acc:.4f} ({cv_acc:.1%})")
        
        if test_acc >= 0.99:
            print("🚨 STILL suspiciously high! Likely synthetic data with perfect rules")
        elif test_acc >= 0.90:
            print("✅ High but reasonable accuracy for a good model")
        else:
            print("✅ Realistic accuracy for real-world data")
    else:
        print("❌ Testing failed")
    
    print("=" * 65)