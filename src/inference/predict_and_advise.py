"""
Updated inference pipeline using geographic-based prediction model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.geographic_predictor import GeographicMicroplasticPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """Engine for making microplastic hotspot predictions using geographic model"""
    
    def __init__(self):
        self.predictor = GeographicMicroplasticPredictor()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the prediction engine with geographic model"""
        try:
            # Initialize the geographic predictor
            success = self.predictor.initialize()
            if success:
                self.is_initialized = True
                logger.info("Geographic prediction engine initialized successfully")
                return True
            else:
                logger.error("Failed to initialize geographic predictor")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing prediction engine: {e}")
            return False
    
    def predict_hotspot(self, latitude: float, longitude: float) -> float:
        """
        Predict hotspot probability for given coordinates
        
        Args:
            latitude: Latitude coordinate (-90 to 90)
            longitude: Longitude coordinate (-180 to 180)
            
        Returns:
            Hotspot probability (0.0 to 1.0)
        """
        if not self.is_initialized:
            raise ValueError("Prediction engine not initialized. Call initialize() first.")
        
        # Validate coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90.")
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180.")
        
        try:
            # Get prediction from geographic model
            probability = self.predictor.predict_hotspot_probability(latitude, longitude)
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error making prediction for ({latitude}, {longitude}): {e}")
            return 0.0  # Default to low risk if prediction fails
    
    def predict_concentration(self, latitude: float, longitude: float) -> float:
        """
        Predict microplastic concentration for given coordinates
        
        Args:
            latitude: Latitude coordinate (-90 to 90)
            longitude: Longitude coordinate (-180 to 180)
            
        Returns:
            Predicted concentration in particles/m³
        """
        if not self.is_initialized:
            raise ValueError("Prediction engine not initialized. Call initialize() first.")
        
        try:
            concentration = self.predictor.predict_concentration(latitude, longitude)
            return float(concentration)
            
        except Exception as e:
            logger.error(f"Error predicting concentration for ({latitude}, {longitude}): {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_info": self.predictor.get_model_info(),
            "prediction_type": "geographic_spatial_interpolation"
        }
