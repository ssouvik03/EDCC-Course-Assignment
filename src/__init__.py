"""
Marine Microplastic Accumulation Zone Prediction Framework
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

from .data import DataLoader, DataPreprocessor
from .features import FeatureEngineerer
from .models import MicroplasticClassifier
from .inference import PredictionEngine, AdvisoryGenerator

__all__ = [
    "DataLoader",
    "DataPreprocessor", 
    "FeatureEngineerer",
    "MicroplasticClassifier",
    "PredictionEngine",
    "AdvisoryGenerator"
]