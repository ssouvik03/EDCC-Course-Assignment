"""
Improved Geographic-Based Microplastic Prediction Model
Uses spatial interpolation and actual data patterns for accurate predictions
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsRegressor
import logging

logger = logging.getLogger(__name__)

class GeographicMicroplasticPredictor:
    """
    Geographic-based microplastic concentration predictor using spatial interpolation
    """
    
    def __init__(self):
        self.data_points = None
        self.knn_model = None
        self.concentration_threshold = 14.1  # Based on our analysis
        self.regional_multipliers = {
            'East Asia (Japan/Korea)': 1.4,      # Higher risk - Tokyo area
            'Indian Ocean/Arabian Sea': 1.3,     # High average concentration  
            'South Atlantic': 1.2,               # Above average
            'China Sea': 1.2,                    # Coastal pollution
            'Arctic': 1.1,                       # Slightly above average
            'US West Coast': 1.0,                # Average
            'North Sea/Europe': 0.9,             # Slightly below average
            'US East Coast': 0.8,                # Below average
            'Australia': 0.6,                    # Lower concentration
            'Other Ocean': 1.0                   # Default
        }
        
    def get_geographic_region(self, lat, lon):
        """Classify coordinates into major geographic regions"""
        if 30 <= lat <= 45 and 130 <= lon <= 150:
            return "East Asia (Japan/Korea)"
        elif 20 <= lat <= 50 and -130 <= lon <= -110:
            return "US West Coast"
        elif 40 <= lat <= 55 and -80 <= lon <= -65:
            return "US East Coast"
        elif 45 <= lat <= 65 and -15 <= lon <= 15:
            return "North Sea/Europe"
        elif 25 <= lat <= 45 and 100 <= lon <= 125:
            return "China Sea"
        elif -40 <= lat <= -25 and 140 <= lon <= 155:
            return "Australia"
        elif 10 <= lat <= 30 and 50 <= lon <= 90:
            return "Indian Ocean/Arabian Sea"
        elif -10 <= lat <= 15 and -60 <= lon <= -30:
            return "South Atlantic"
        elif 50 <= lat <= 80 and -180 <= lon <= 180:
            return "Arctic"
        else:
            return "Other Ocean"
    
    def initialize(self, data_path='data/marine_microplastics.csv'):
        """Initialize the model with actual data points"""
        try:
            # Load the actual microplastic data
            df = pd.read_csv(data_path)
            
            # Store coordinates and concentrations
            self.data_points = df[['Latitude', 'Longitude', 'Concentration']].copy()
            
            # Add geographic regions
            self.data_points['region'] = self.data_points.apply(
                lambda row: self.get_geographic_region(row['Latitude'], row['Longitude']), 
                axis=1
            )
            
            # Initialize K-Nearest Neighbors for spatial interpolation
            coords = self.data_points[['Latitude', 'Longitude']].values
            concentrations = self.data_points['Concentration'].values
            
            # Use more neighbors for smoother interpolation
            self.knn_model = KNeighborsRegressor(n_neighbors=10, weights='distance')
            self.knn_model.fit(coords, concentrations)
            
            logger.info(f"Geographic predictor initialized with {len(self.data_points)} data points")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize geographic predictor: {e}")
            return False
    
    def predict_concentration(self, lat, lon):
        """
        Predict microplastic concentration for given coordinates
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            
        Returns:
            Predicted concentration in particles/m³
        """
        if self.knn_model is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Get basic prediction from spatial interpolation
        coords = np.array([[lat, lon]])
        base_concentration = self.knn_model.predict(coords)[0]
        
        # Apply regional adjustment
        region = self.get_geographic_region(lat, lon)
        regional_multiplier = self.regional_multipliers.get(region, 1.0)
        
        # Calculate distance to nearest high-concentration point
        high_conc_points = self.data_points[self.data_points['Concentration'] > 30]
        if len(high_conc_points) > 0:
            high_coords = high_conc_points[['Latitude', 'Longitude']].values
            distances = cdist([[lat, lon]], high_coords)[0]
            min_distance = np.min(distances)
            
            # Apply proximity bonus (closer to known hotspots = higher risk)
            if min_distance < 5:  # Within 5 degrees
                proximity_multiplier = 1.2
            elif min_distance < 10:  # Within 10 degrees
                proximity_multiplier = 1.1
            else:
                proximity_multiplier = 1.0
        else:
            proximity_multiplier = 1.0
        
        # Final adjusted concentration
        adjusted_concentration = base_concentration * regional_multiplier * proximity_multiplier
        
        # Ensure realistic bounds
        adjusted_concentration = max(0.5, min(adjusted_concentration, 250))
        
        return adjusted_concentration
    
    def predict_hotspot_probability(self, lat, lon):
        """
        Predict hotspot probability for given coordinates
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            
        Returns:
            Probability of being a microplastic hotspot (0.0 to 1.0)
        """
        concentration = self.predict_concentration(lat, lon)
        
        # Convert concentration to probability using sigmoid function
        # Adjusted to be more sensitive around the threshold
        if concentration >= 50:
            probability = 0.9  # Very high concentration
        elif concentration >= 30:
            probability = 0.7 + (concentration - 30) * 0.01  # High concentration
        elif concentration >= 20:
            probability = 0.5 + (concentration - 20) * 0.02  # Medium-high
        elif concentration >= 14.1:  # Our threshold
            probability = 0.3 + (concentration - 14.1) * 0.034  # Above threshold
        elif concentration >= 10:
            probability = 0.15 + (concentration - 10) * 0.037  # Below threshold
        else:
            probability = concentration * 0.015  # Very low
        
        # Ensure probability bounds
        probability = max(0.0, min(probability, 1.0))
        
        return probability
    
    def get_model_info(self):
        """Get information about the model"""
        if self.data_points is None:
            return "Model not initialized"
        
        info = {
            'model_type': 'Geographic K-Nearest Neighbors',
            'data_points': len(self.data_points),
            'threshold': self.concentration_threshold,
            'regions': len(self.regional_multipliers),
            'interpolation_neighbors': 10
        }
        return info

# Create global instance for easy import
geographic_predictor = GeographicMicroplasticPredictor()