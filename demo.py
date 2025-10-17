#!/usr/bin/env python3
"""
Marine Microplastic Prediction Framework - Quick Start Demo
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data.preprocess import DataLoader, DataPreprocessor
from features.engineer_features import FeatureEngineerer
from inference.predict_and_advise import AdvisoryGenerator

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic marine microplastic data
    data = {
        'Date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'Latitude': np.random.uniform(-60, 60, n_samples),
        'Longitude': np.random.uniform(-180, 180, n_samples),
        'Oceans': np.random.choice(['Pacific', 'Atlantic', 'Indian', 'Arctic'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Regions': np.random.choice(['North', 'South', 'Equatorial', 'Polar'], n_samples),
        'Concentration': np.random.lognormal(2, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlations to make it more realistic
    # Higher concentrations near certain coordinates (simulate gyres)
    gyre_locations = [(35, -140), (30, -40), (-30, 0)]  # Pacific, Atlantic, Indian gyres
    
    for i, (lat, lon) in enumerate(gyre_locations):
        distance = np.sqrt((df['Latitude'] - lat)**2 + (df['Longitude'] - lon)**2)
        proximity_factor = np.exp(-distance / 30)  # Exponential decay with distance
        df['Concentration'] += proximity_factor * np.random.normal(2, 0.5, n_samples)
    
    # Ensure concentrations are positive
    df['Concentration'] = np.abs(df['Concentration'])
    
    return df

def main():
    """Main demonstration function"""
    print("🌊 MARINE MICROPLASTIC PREDICTION FRAMEWORK")
    print("=" * 50)
    print("Quick Start Demonstration")
    print()
    
    # Step 1: Create sample data
    print("📊 Step 1: Creating sample dataset...")
    df = create_sample_dataset()
    print(f"   ✅ Created dataset with {len(df):,} records")
    print(f"   📅 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   🌍 Coordinate range: Lat [{df['Latitude'].min():.1f}°, {df['Latitude'].max():.1f}°], Lon [{df['Longitude'].min():.1f}°, {df['Longitude'].max():.1f}°]")
    print()
    
    # Step 2: Data preprocessing
    print("🔧 Step 2: Data preprocessing...")
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Validate schema
    if not loader.validate_data_schema(df):
        print("   ❌ Data schema validation failed")
        return
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    print(f"   ✅ Data cleaned: {len(df_clean):,} records retained")
    
    # Create target variable
    df_processed = preprocessor.create_target_variable(df_clean, concentration_col='Concentration')
    hotspot_ratio = df_processed['is_hotspot'].mean()
    print(f"   ✅ Target variable created: {hotspot_ratio:.1%} hotspot ratio")
    print()
    
    # Step 3: Feature engineering
    print("⚙️ Step 3: Feature engineering...")
    engineer = FeatureEngineerer()
    df_features = engineer.create_all_features(df_processed)
    
    original_features = len(df_processed.columns)
    total_features = len(df_features.columns)
    new_features = total_features - original_features
    
    print(f"   ✅ Feature engineering complete:")
    print(f"      📈 Original features: {original_features}")
    print(f"      🚀 New features created: {new_features}")
    print(f"      📊 Total features: {total_features}")
    print()
    
    # Step 4: Feature importance groups
    print("📋 Step 4: Feature categorization...")
    feature_groups = engineer.get_feature_importance_groups()
    for group_name, features in feature_groups.items():
        print(f"   🏷️ {group_name.title()}: {len(features)} features")
    print()
    
    # Step 5: Sample predictions and advisories
    print("🔮 Step 5: Generating sample predictions and advisories...")
    advisory_generator = AdvisoryGenerator()
    
    # Test locations
    test_locations = [
        {
            'latitude': 35.0, 'longitude': -140.0, 'ocean': 'Pacific', 'region': 'North Pacific Gyre',
            'probability': 0.85, 'risk': 'Very High Risk'
        },
        {
            'latitude': 40.7, 'longitude': -74.0, 'ocean': 'Atlantic', 'region': 'New York Bight',
            'probability': 0.45, 'risk': 'Moderate Risk'
        },
        {
            'latitude': -20.0, 'longitude': 150.0, 'ocean': 'Pacific', 'region': 'Coral Sea',
            'probability': 0.15, 'risk': 'Low Risk'
        }
    ]
    
    print("   📍 Sample prediction locations:")
    for i, location in enumerate(test_locations, 1):
        print(f"      {i}. {location['latitude']:.1f}°, {location['longitude']:.1f}° ({location['ocean']}) - {location['probability']:.0%} risk")
    print()
    
    # Generate sample advisories
    print("📄 Step 6: Generating stakeholder advisories...")
    
    for i, location in enumerate(test_locations[:2], 1):  # Just show first 2
        # Mock prediction result
        prediction_result = {
            'location': {
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'ocean': location['ocean'],
                'region': location['region']
            },
            'prediction_date': '2024-10-17',
            'ensemble_result': {
                'is_hotspot': location['probability'] > 0.5,
                'hotspot_probability': location['probability'],
                'confidence': 'High' if location['probability'] > 0.7 or location['probability'] < 0.3 else 'Medium'
            },
            'confidence_level': location['risk']
        }
        
        # Generate advisory for public
        advisory = advisory_generator.generate_advisory(
            prediction_result, stakeholder='public', use_llm=False
        )
        
        print(f"   📋 Location {i} Advisory (Public):")
        print(f"      🎯 Risk Level: {location['risk']}")
        print(f"      📊 Probability: {location['probability']:.0%}")
        print(f"      📝 Advisory ID: {advisory['metadata']['advisory_id']}")
        print(f"      🏷️ Recommendations: {len(advisory['recommendations'])} items")
        print()
    
    # Step 7: Data export
    print("💾 Step 7: Saving processed data...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save datasets
    df_processed.to_csv('data/processed_demo_data.csv', index=False)
    df_features.to_csv('data/features_demo_data.csv', index=False)
    
    print(f"   ✅ Saved processed data: data/processed_demo_data.csv")
    print(f"   ✅ Saved feature data: data/features_demo_data.csv")
    print()
    
    # Final summary
    print("🎉 DEMONSTRATION COMPLETE!")
    print("=" * 50)
    print("✅ Framework Status:")
    print("   🔧 Data processing: Operational")
    print("   ⚙️ Feature engineering: Operational") 
    print("   📊 Advisory generation: Operational")
    print("   💾 Data export: Complete")
    print()
    print("📋 Next Steps:")
    print("   1. Download real marine microplastic dataset from Kaggle")
    print("   2. Run full model training: python src/models/train_classifier.py")
    print("   3. Launch Jupyter notebooks for analysis: jupyter notebook notebooks/")
    print("   4. Deploy inference pipeline for real-time predictions")
    print()
    print("🔗 Key Files:")
    print("   📖 README.md - Project documentation")
    print("   📓 notebooks/ - Interactive analysis notebooks")
    print("   ⚙️ config/ - Configuration files")
    print("   🧪 tests/ - Test suite")
    print()
    print("Thank you for using the Marine Microplastic Prediction Framework! 🌊")

if __name__ == "__main__":
    main()