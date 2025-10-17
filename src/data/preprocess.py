"""
Data loading and preprocessing for marine microplastic analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate marine microplastic data"""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
        
    def load_raw_data(self, filename: str = "marine_microplastics.csv") -> pd.DataFrame:
        """
        Load raw marine microplastic data
        
        Args:
            filename: Name of the data file
            
        Returns:
            DataFrame with raw data
        """
        filepath = self.data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that required columns exist
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if schema is valid
        """
        required_columns = ['Latitude', 'Longitude', 'Date', 'Oceans', 'Regions']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        logger.info("Data schema validation passed")
        return True


class DataPreprocessor:
    """Preprocess and clean marine microplastic data"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataset
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_len - len(df_clean)} duplicate records")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Validate coordinates
        df_clean = self._validate_coordinates(df_clean)
        
        # Parse dates
        df_clean = self._parse_dates(df_clean)
        
        # Clean categorical variables
        df_clean = self._clean_categorical(df_clean)
        
        logger.info(f"Data cleaning complete. Final dataset: {len(df_clean)} records")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        
        # Drop rows with missing coordinates
        df_clean = df_clean.dropna(subset=['Latitude', 'Longitude'])
        
        # Fill missing dates with median date
        if df_clean['Date'].isnull().any():
            df_clean['Date'] = df_clean['Date'].fillna(df_clean['Date'].mode()[0])
        
        # Fill missing categorical variables with 'Unknown'
        categorical_cols = ['Oceans', 'Regions']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        return df_clean
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean coordinate data"""
        df_clean = df.copy()
        
        # Remove invalid coordinates
        valid_lat = (df_clean['Latitude'] >= -90) & (df_clean['Latitude'] <= 90)
        valid_lon = (df_clean['Longitude'] >= -180) & (df_clean['Longitude'] <= 180)
        
        invalid_coords = ~(valid_lat & valid_lon)
        if invalid_coords.any():
            logger.warning(f"Removing {invalid_coords.sum()} records with invalid coordinates")
            df_clean = df_clean[valid_lat & valid_lon]
        
        return df_clean
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and standardize date formats"""
        df_clean = df.copy()
        
        try:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            
            # Remove records with invalid dates
            invalid_dates = df_clean['Date'].isnull()
            if invalid_dates.any():
                logger.warning(f"Removing {invalid_dates.sum()} records with invalid dates")
                df_clean = df_clean[~invalid_dates]
                
        except Exception as e:
            logger.error(f"Error parsing dates: {e}")
            raise
        
        return df_clean
    
    def _clean_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize categorical variables"""
        df_clean = df.copy()
        
        categorical_cols = ['Oceans', 'Regions']
        
        for col in categorical_cols:
            if col in df_clean.columns:
                # Strip whitespace and standardize case
                df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        
        return df_clean
    
    def create_target_variable(self, df: pd.DataFrame, 
                             concentration_col: str = 'Concentration',
                             threshold_percentile: float = 75) -> pd.DataFrame:
        """
        Create binary target variable for microplastic hotspot classification
        
        Args:
            df: Input DataFrame
            concentration_col: Column name for microplastic concentration
            threshold_percentile: Percentile threshold for high/low classification
            
        Returns:
            DataFrame with target variable
        """
        df_target = df.copy()
        
        if concentration_col not in df_target.columns:
            logger.warning(f"Concentration column '{concentration_col}' not found. Creating synthetic target.")
            # Create synthetic target based on location clustering (for demonstration)
            df_target['is_hotspot'] = self._create_synthetic_target(df_target)
        else:
            # Use concentration threshold
            threshold = np.percentile(df_target[concentration_col].dropna(), threshold_percentile)
            df_target['is_hotspot'] = (df_target[concentration_col] >= threshold).astype(int)
        
        logger.info(f"Created target variable. Hotspot ratio: {df_target['is_hotspot'].mean():.3f}")
        return df_target
    
    def _create_synthetic_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create synthetic target variable based on spatial patterns"""
        from sklearn.cluster import KMeans
        
        # Use K-means clustering on coordinates to create synthetic hotspots
        coords = df[['Latitude', 'Longitude']].values
        kmeans = KMeans(n_clusters=8, random_state=42)
        clusters = kmeans.fit_predict(coords)
        
        # Assign certain clusters as hotspots (e.g., clusters with higher density)
        cluster_counts = pd.Series(clusters).value_counts()
        hotspot_clusters = cluster_counts.index[:3]  # Top 3 densest clusters
        
        is_hotspot = np.isin(clusters, hotspot_clusters).astype(int)
        return is_hotspot


def main():
    """Main function for data preprocessing"""
    # Initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    try:
        # Load data
        df = loader.load_raw_data()
        
        # Validate schema
        if not loader.validate_data_schema(df):
            raise ValueError("Data schema validation failed")
        
        # Clean data
        df_clean = preprocessor.clean_data(df)
        
        # Create target variable
        df_final = preprocessor.create_target_variable(df_clean)
        
        # Save processed data
        output_path = Path("data/processed_marine_microplastics.csv")
        df_final.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        # Print summary statistics
        print("\n=== Data Processing Summary ===")
        print(f"Total records: {len(df_final)}")
        print(f"Features: {df_final.columns.tolist()}")
        print(f"Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
        print(f"Coordinate bounds: Lat[{df_final['Latitude'].min():.2f}, {df_final['Latitude'].max():.2f}], "
              f"Lon[{df_final['Longitude'].min():.2f}, {df_final['Longitude'].max():.2f}]")
        print(f"Hotspot distribution: {df_final['is_hotspot'].value_counts().to_dict()}")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()