"""
Test suite for the marine microplastic prediction framework
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocess import DataLoader, DataPreprocessor
from features.engineer_features import FeatureEngineerer
from models.train_classifier import MicroplasticClassifier
from inference.predict_and_advise import PredictionEngine, AdvisoryGenerator


class TestDataProcessing(unittest.TestCase):
    """Test data loading and preprocessing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'Latitude': np.random.uniform(-60, 60, 100),
            'Longitude': np.random.uniform(-180, 180, 100),
            'Oceans': np.random.choice(['Pacific', 'Atlantic', 'Indian'], 100),
            'Regions': np.random.choice(['North', 'South', 'Equatorial'], 100),
            'Concentration': np.random.lognormal(2, 1, 100)
        })
    
    def test_data_schema_validation(self):
        """Test data schema validation"""
        # Valid schema should pass
        self.assertTrue(self.loader.validate_data_schema(self.sample_data))
        
        # Missing required column should fail
        invalid_data = self.sample_data.drop('Latitude', axis=1)
        self.assertFalse(self.loader.validate_data_schema(invalid_data))
    
    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        # Add some problematic data
        dirty_data = self.sample_data.copy()
        dirty_data.loc[0, 'Latitude'] = 91  # Invalid latitude
        dirty_data.loc[1, 'Longitude'] = -181  # Invalid longitude
        dirty_data.loc[2, 'Date'] = None  # Missing date
        
        cleaned_data = self.preprocessor.clean_data(dirty_data)
        
        # Should remove invalid coordinates
        self.assertTrue(all(cleaned_data['Latitude'].between(-90, 90)))
        self.assertTrue(all(cleaned_data['Longitude'].between(-180, 180)))
        
        # Should handle missing dates
        self.assertFalse(cleaned_data['Date'].isnull().any())
    
    def test_target_variable_creation(self):
        """Test target variable creation"""
        result_data = self.preprocessor.create_target_variable(self.sample_data)
        
        # Should have target variable
        self.assertIn('is_hotspot', result_data.columns)
        
        # Should be binary
        unique_values = result_data['is_hotspot'].unique()
        self.assertTrue(all(val in [0, 1] for val in unique_values))


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engineer = FeatureEngineerer()
        
        # Create sample data with required columns
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=50, freq='D'),
            'Latitude': np.random.uniform(-60, 60, 50),
            'Longitude': np.random.uniform(-180, 180, 50),
            'Oceans': np.random.choice(['Pacific', 'Atlantic', 'Indian'], 50),
            'Regions': np.random.choice(['North', 'South', 'Equatorial'], 50),
            'is_hotspot': np.random.choice([0, 1], 50)
        })
    
    def test_temporal_features(self):
        """Test temporal feature creation"""
        result = self.engineer._create_temporal_features(self.sample_data)
        
        # Should have temporal features
        expected_features = ['year', 'month', 'day', 'quarter', 'month_sin', 'month_cos']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
    
    def test_spatial_features(self):
        """Test spatial feature creation"""
        result = self.engineer._create_spatial_features(self.sample_data)
        
        # Should have spatial features
        expected_features = ['lat_rad', 'lon_rad', 'x', 'y', 'z', 'distance_from_equator']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
    
    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline"""
        original_columns = len(self.sample_data.columns)
        result = self.engineer.create_all_features(self.sample_data)
        
        # Should create additional features
        self.assertGreater(len(result.columns), original_columns)
        
        # Should preserve original data
        self.assertEqual(len(result), len(self.sample_data))


class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = MicroplasticClassifier()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.choice([0, 1], n_samples)
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
    
    def test_data_preparation(self):
        """Test data preparation for training"""
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df['is_hotspot'] = self.y
        
        X, y, features = self.classifier.prepare_data(df, target_col='is_hotspot')
        
        self.assertEqual(X.shape[0], len(self.y))
        self.assertEqual(X.shape[1], len(self.feature_names))
        self.assertEqual(len(features), len(self.feature_names))
    
    def test_model_training(self):
        """Test model training process"""
        metrics = self.classifier.train(self.X, self.y, self.feature_names)
        
        # Should have trained models
        self.assertTrue(self.classifier.is_trained)
        self.assertIn('individual_models', metrics)
        self.assertIn('cross_validation', metrics)
    
    def test_predictions(self):
        """Test model predictions"""
        # Train model first
        self.classifier.train(self.X, self.y, self.feature_names)
        
        # Test predictions
        X_test = np.random.randn(10, len(self.feature_names))
        predictions = self.classifier.predict(X_test)
        
        # Should have predictions for each model
        self.assertIn('random_forest', predictions)
        self.assertIn('predictions', predictions['random_forest'])


class TestInferencePipeline(unittest.TestCase):
    """Test inference and advisory generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.prediction_engine = PredictionEngine()
        self.advisory_generator = AdvisoryGenerator()
    
    def test_advisory_generation(self):
        """Test advisory generation"""
        # Mock prediction result
        prediction_result = {
            'location': {
                'latitude': 35.0,
                'longitude': -140.0,
                'ocean': 'Pacific',
                'region': 'North Pacific'
            },
            'prediction_date': '2024-03-15',
            'ensemble_result': {
                'is_hotspot': True,
                'hotspot_probability': 0.75,
                'confidence': 'High'
            },
            'confidence_level': 'High Risk'
        }
        
        advisory = self.advisory_generator.generate_advisory(
            prediction_result, stakeholder='public', use_llm=False
        )
        
        # Should have required fields
        self.assertIn('advisory_text', advisory)
        self.assertIn('metadata', advisory)
        self.assertIn('recommendations', advisory)
    
    def test_risk_level_determination(self):
        """Test risk level classification"""
        # High risk
        high_risk = self.advisory_generator._determine_risk_level(0.8)
        self.assertEqual(high_risk, 'high_risk')
        
        # Moderate risk
        moderate_risk = self.advisory_generator._determine_risk_level(0.5)
        self.assertEqual(moderate_risk, 'moderate_risk')
        
        # Low risk
        low_risk = self.advisory_generator._determine_risk_level(0.2)
        self.assertEqual(low_risk, 'low_risk')


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_haversine_distance(self):
        """Test haversine distance calculation"""
        engineer = FeatureEngineerer()
        
        # Distance between known points (NYC to LA approximately 3944 km)
        distance = engineer._haversine_distance(40.7, -74.0, 34.0, -118.2)
        
        # Should be approximately correct (within 100km tolerance)
        self.assertAlmostEqual(distance, 3944, delta=100)
    
    def test_coordinate_validation(self):
        """Test coordinate validation"""
        preprocessor = DataPreprocessor()
        
        test_data = pd.DataFrame({
            'Latitude': [-90, 0, 90, 95, -95],
            'Longitude': [-180, 0, 180, 185, -185]
        })
        
        validated = preprocessor._validate_coordinates(test_data)
        
        # Should keep only valid coordinates
        self.assertEqual(len(validated), 3)
        self.assertTrue(all(validated['Latitude'].between(-90, 90)))
        self.assertTrue(all(validated['Longitude'].between(-180, 180)))


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestDataProcessing,
        TestFeatureEngineering, 
        TestModelTraining,
        TestInferencePipeline,
        TestUtilityFunctions
    ]
    
    loader = unittest.TestLoader()
    suites = [loader.loadTestsFromTestCase(test_class) for test_class in test_classes]
    combined_suite = unittest.TestSuite(suites)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split()[0] if 'AssertionError' in traceback else 'Unknown error'}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split()[-1] if traceback else 'Unknown error'}")