"""
Feature engineering for marine microplastic spatio-temporal analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineerer:
    """Engineer features for microplastic hotspot prediction"""
    
    def __init__(self):
        self.feature_names = []
        self.encoders = {}
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for microplastic prediction
        
        Args:
            df: Input DataFrame with cleaned data
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Temporal features
        df_features = self._create_temporal_features(df_features)
        
        # Spatial features
        df_features = self._create_spatial_features(df_features)
        
        # Oceanographic features
        df_features = self._create_oceanographic_features(df_features)
        
        # Distance-based features
        df_features = self._create_distance_features(df_features)
        
        # Interaction features
        df_features = self._create_interaction_features(df_features)
        
        # Statistical aggregation features
        df_features = self._create_aggregation_features(df_features)
        
        logger.info(f"Feature engineering complete. Total features: {len(df_features.columns)}")
        return df_features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date information"""
        df_temp = df.copy()
        
        # Extract basic temporal components
        df_temp['year'] = df_temp['Date'].dt.year
        df_temp['month'] = df_temp['Date'].dt.month
        df_temp['day'] = df_temp['Date'].dt.day
        df_temp['day_of_year'] = df_temp['Date'].dt.dayofyear
        df_temp['week_of_year'] = df_temp['Date'].dt.isocalendar().week
        df_temp['quarter'] = df_temp['Date'].dt.quarter
        
        # Seasonal features
        df_temp['season'] = df_temp['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # Cyclical encoding for temporal features
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        df_temp['day_sin'] = np.sin(2 * np.pi * df_temp['day_of_year'] / 365)
        df_temp['day_cos'] = np.cos(2 * np.pi * df_temp['day_of_year'] / 365)
        
        # Time-based trend features
        reference_date = df_temp['Date'].min()
        df_temp['days_since_start'] = (df_temp['Date'] - reference_date).dt.days
        df_temp['years_since_start'] = df_temp['days_since_start'] / 365.25
        
        logger.info("Created temporal features")
        return df_temp
    
    def _create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features from coordinates"""
        df_spatial = df.copy()
        
        # Basic coordinate transformations
        df_spatial['lat_rad'] = np.radians(df_spatial['Latitude'])
        df_spatial['lon_rad'] = np.radians(df_spatial['Longitude'])
        
        # Cartesian coordinates (for easier distance calculations)
        R = 6371  # Earth radius in km
        df_spatial['x'] = R * np.cos(df_spatial['lat_rad']) * np.cos(df_spatial['lon_rad'])
        df_spatial['y'] = R * np.cos(df_spatial['lat_rad']) * np.sin(df_spatial['lon_rad'])
        df_spatial['z'] = R * np.sin(df_spatial['lat_rad'])
        
        # Spatial bins/grids
        df_spatial['lat_bin'] = pd.cut(df_spatial['Latitude'], bins=20, labels=False)
        df_spatial['lon_bin'] = pd.cut(df_spatial['Longitude'], bins=20, labels=False)
        df_spatial['spatial_grid'] = df_spatial['lat_bin'] * 20 + df_spatial['lon_bin']
        
        # Distance from equator and prime meridian
        df_spatial['distance_from_equator'] = np.abs(df_spatial['Latitude'])
        df_spatial['distance_from_prime_meridian'] = np.abs(df_spatial['Longitude'])
        
        # Hemisphere indicators
        df_spatial['northern_hemisphere'] = (df_spatial['Latitude'] >= 0).astype(int)
        df_spatial['eastern_hemisphere'] = (df_spatial['Longitude'] >= 0).astype(int)
        
        # Coastal proximity proxy (simplified)
        df_spatial['coastal_proxy'] = self._calculate_coastal_proximity(df_spatial)
        
        logger.info("Created spatial features")
        return df_spatial
    
    def _create_oceanographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from oceanographic context"""
        df_ocean = df.copy()
        
        # Encode categorical ocean and region variables
        df_ocean = self._encode_categorical_features(df_ocean, ['Oceans', 'Regions'])
        
        # Ocean-specific features
        major_oceans = ['Pacific', 'Atlantic', 'Indian', 'Arctic', 'Southern']
        for ocean in major_oceans:
            df_ocean[f'is_{ocean.lower()}'] = df_ocean['Oceans'].str.contains(ocean, case=False, na=False).astype(int)
        
        # Create ocean basin features (simplified regional grouping)
        df_ocean['ocean_basin'] = self._assign_ocean_basin(df_ocean)
        
        # Water body type classification
        df_ocean['water_body_type'] = self._classify_water_body_type(df_ocean)
        
        logger.info("Created oceanographic features")
        return df_ocean
    
    def _create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distance-based features"""
        df_dist = df.copy()
        
        # Distance to major reference points
        reference_points = {
            'pacific_gyre': (35.0, -140.0),  # North Pacific Gyre
            'atlantic_gyre': (30.0, -40.0),  # North Atlantic Gyre
            'great_barrier_reef': (-18.0, 147.0),
            'mediterranean_center': (36.0, 15.0),
            'north_pole': (90.0, 0.0),
            'equator_center': (0.0, 0.0)
        }
        
        for point_name, (lat, lon) in reference_points.items():
            df_dist[f'distance_to_{point_name}'] = self._haversine_distance(
                df_dist['Latitude'], df_dist['Longitude'], lat, lon
            )
        
        # Distance to nearest data point (local density proxy)
        df_dist['nearest_neighbor_distance'] = self._calculate_nearest_neighbor_distance(df_dist)
        
        # Distance to centroid of dataset
        centroid_lat = df_dist['Latitude'].mean()
        centroid_lon = df_dist['Longitude'].mean()
        df_dist['distance_to_centroid'] = self._haversine_distance(
            df_dist['Latitude'], df_dist['Longitude'], centroid_lat, centroid_lon
        )
        
        logger.info("Created distance-based features")
        return df_dist
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different feature types"""
        df_interact = df.copy()
        
        # Temporal-spatial interactions
        df_interact['lat_month_interaction'] = df_interact['Latitude'] * df_interact['month']
        df_interact['lon_season_interaction'] = df_interact['Longitude'] * df_interact['quarter']
        
        # Distance-temporal interactions
        if 'distance_to_pacific_gyre' in df_interact.columns:
            df_interact['gyre_distance_season'] = (
                df_interact['distance_to_pacific_gyre'] * df_interact['quarter']
            )
        
        # Coordinate products
        df_interact['lat_lon_product'] = df_interact['Latitude'] * df_interact['Longitude']
        df_interact['coord_magnitude'] = np.sqrt(df_interact['Latitude']**2 + df_interact['Longitude']**2)
        
        logger.info("Created interaction features")
        return df_interact
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features"""
        df_agg = df.copy()
        
        # Spatial aggregations (by grid cells)
        if 'spatial_grid' in df_agg.columns:
            spatial_stats = df_agg.groupby('spatial_grid').agg({
                'Latitude': ['count', 'mean', 'std'],
                'Longitude': ['mean', 'std'],
                'year': ['min', 'max', 'nunique']
            }).fillna(0)
            
            # Flatten column names
            spatial_stats.columns = ['_'.join(col).strip() for col in spatial_stats.columns]
            spatial_stats = spatial_stats.add_prefix('spatial_')
            
            # Merge back to main dataframe
            df_agg = df_agg.merge(spatial_stats, left_on='spatial_grid', right_index=True, how='left')
        
        # Temporal aggregations (by month)
        monthly_stats = df_agg.groupby('month').agg({
            'Latitude': ['count', 'mean'],
            'Longitude': ['mean'],
        }).fillna(0)
        
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
        monthly_stats = monthly_stats.add_prefix('monthly_')
        
        df_agg = df_agg.merge(monthly_stats, left_on='month', right_index=True, how='left')
        
        logger.info("Created aggregation features")
        return df_agg
    
    def _encode_categorical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features using multiple strategies"""
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                # Label encoding
                unique_values = df_encoded[col].unique()
                label_map = {val: idx for idx, val in enumerate(unique_values)}
                df_encoded[f'{col}_encoded'] = df_encoded[col].map(label_map)
                
                # One-hot encoding for top categories
                top_categories = df_encoded[col].value_counts().head(5).index
                for category in top_categories:
                    df_encoded[f'{col}_{category}'] = (df_encoded[col] == category).astype(int)
                
                # Store encoder for later use
                self.encoders[col] = label_map
        
        return df_encoded
    
    def _calculate_coastal_proximity(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate simplified coastal proximity measure"""
        # Simplified heuristic: distance from major coastal coordinates
        # In practice, this would use actual coastline data
        coastal_points = [
            (40.7, -74.0),  # New York coast
            (34.0, -118.2), # Los Angeles coast
            (51.5, 0.1),    # London coast
            (35.7, 139.7),  # Tokyo coast
            (-33.9, 151.2), # Sydney coast
        ]
        
        min_distances = []
        for _, row in df.iterrows():
            distances = [
                self._haversine_distance(row['Latitude'], row['Longitude'], lat, lon)
                for lat, lon in coastal_points
            ]
            min_distances.append(min(distances))
        
        return np.array(min_distances)
    
    def _assign_ocean_basin(self, df: pd.DataFrame) -> pd.Series:
        """Assign ocean basin based on coordinates"""
        def get_basin(lat, lon):
            if -180 <= lon <= -30:
                return 'Atlantic_Basin'
            elif -30 < lon <= 20:
                return 'Atlantic_Basin' if lat > -35 else 'Southern_Basin'
            elif 20 < lon <= 147:
                return 'Indian_Basin'
            else:
                return 'Pacific_Basin'
        
        return df.apply(lambda row: get_basin(row['Latitude'], row['Longitude']), axis=1)
    
    def _classify_water_body_type(self, df: pd.DataFrame) -> pd.Series:
        """Classify water body type based on location"""
        def get_water_type(lat, lon):
            if abs(lat) > 66.5:
                return 'Polar'
            elif abs(lat) < 23.5:
                return 'Tropical'
            else:
                return 'Temperate'
        
        return df.apply(lambda row: get_water_type(row['Latitude'], row['Longitude']), axis=1)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_nearest_neighbor_distance(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate distance to nearest neighbor for each point"""
        from sklearn.neighbors import NearestNeighbors
        
        coords = df[['Latitude', 'Longitude']].values
        
        # Handle case where there's only one data point
        if len(coords) < 2:
            return np.array([0.0] * len(coords))
        
        # Use minimum of available neighbors and 2
        n_neighbors = min(2, len(coords))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='haversine').fit(np.radians(coords))
        distances, _ = nbrs.kneighbors(np.radians(coords))
        
        # Return distance to nearest neighbor (excluding self)
        if distances.shape[1] > 1:
            return distances[:, 1] * 6371  # Convert to km
        else:
            return np.array([0.0] * len(coords))
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by type for analysis"""
        feature_groups = {
            'temporal': [col for col in self.feature_names if any(x in col.lower() for x in 
                        ['year', 'month', 'day', 'season', 'quarter', 'sin', 'cos', 'since'])],
            'spatial': [col for col in self.feature_names if any(x in col.lower() for x in 
                       ['lat', 'lon', 'x', 'y', 'z', 'bin', 'grid', 'hemisphere', 'equator'])],
            'oceanographic': [col for col in self.feature_names if any(x in col.lower() for x in 
                             ['ocean', 'region', 'basin', 'water'])],
            'distance': [col for col in self.feature_names if 'distance' in col.lower()],
            'interaction': [col for col in self.feature_names if 'interaction' in col.lower() or 
                           'product' in col.lower() or 'magnitude' in col.lower()],
            'aggregation': [col for col in self.feature_names if any(x in col.lower() for x in 
                           ['spatial_', 'monthly_', 'count', 'mean', 'std'])]
        }
        
        return feature_groups


def main():
    """Main function for feature engineering"""
    engineer = FeatureEngineerer()
    
    try:
        # Load processed data
        data_path = Path("data/processed_marine_microplastics.csv")
        if not data_path.exists():
            raise FileNotFoundError("Processed data not found. Run data preprocessing first.")
        
        df = pd.read_csv(data_path, parse_dates=['Date'])
        logger.info(f"Loaded {len(df)} records for feature engineering")
        
        # Create features
        df_features = engineer.create_all_features(df)
        
        # Store feature names
        engineer.feature_names = df_features.columns.tolist()
        
        # Save feature-engineered data
        output_path = Path("data/features_marine_microplastics.csv")
        df_features.to_csv(output_path, index=False)
        logger.info(f"Feature-engineered data saved to {output_path}")
        
        # Print feature summary
        print("\n=== Feature Engineering Summary ===")
        print(f"Total features created: {len(df_features.columns)}")
        
        feature_groups = engineer.get_feature_importance_groups()
        for group_name, features in feature_groups.items():
            print(f"{group_name.title()} features: {len(features)}")
        
        print(f"\nDataset shape: {df_features.shape}")
        print(f"Memory usage: {df_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()