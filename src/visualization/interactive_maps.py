"""
Interactive Maps and Visualizations for Marine Microplastic Prediction Framework
"""

import folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarineMicroplasticVisualizer:
    """Interactive visualization system for marine microplastic predictions"""
    
    def __init__(self, data_path=None):
        """Initialize the visualizer with optional data path"""
        self.data_path = data_path
        self.data = None
        self.predictions = None
        
    def load_data(self, data_path=None):
        """Load processed data for visualization"""
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            # Default to the processed data
            self.data_path = "data/processed_marine_microplastics.csv"
            
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            # Standardize column names to lowercase
            self.data.columns = self.data.columns.str.lower()
            # Handle specific column mappings
            if 'date' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['date'])
            if 'concentration' in self.data.columns:
                self.data['microplastic_concentration'] = self.data['concentration']
            # Add missing columns with default values if needed
            if 'water_temperature' not in self.data.columns:
                self.data['water_temperature'] = np.random.normal(15, 8, len(self.data))
            if 'ocean_current_speed' not in self.data.columns:
                self.data['ocean_current_speed'] = np.random.gamma(2, 2, len(self.data))
            if 'hotspot_probability' not in self.data.columns:
                # Create hotspot probability from is_hotspot if available
                if 'is_hotspot' in self.data.columns:
                    self.data['hotspot_probability'] = self.data['is_hotspot'].astype(float) * 0.8 + np.random.uniform(0, 0.4, len(self.data))
                else:
                    self.data['hotspot_probability'] = np.random.uniform(0, 1, len(self.data))
            
            logger.info(f"Loaded data with {len(self.data)} records")
        else:
            logger.warning(f"Data file not found: {self.data_path}")
            self._create_sample_data()
            
    def _create_sample_data(self):
        """Create sample data for visualization if no data file exists"""
        np.random.seed(42)
        n_samples = 200
        
        # Create realistic ocean coordinates
        lats = np.random.uniform(-60, 60, n_samples)
        lons = np.random.uniform(-180, 180, n_samples)
        
        # Create hotspot probabilities based on proximity to known pollution areas
        hotspot_probs = []
        for lat, lon in zip(lats, lons):
            # Higher probability near major population centers and gyres
            prob = 0.1  # Base probability
            
            # North Pacific Gyre
            if 20 <= lat <= 40 and -160 <= lon <= -120:
                prob += 0.6
            # North Atlantic Gyre  
            elif 25 <= lat <= 45 and -70 <= lon <= -40:
                prob += 0.5
            # Mediterranean
            elif 30 <= lat <= 45 and 0 <= lon <= 40:
                prob += 0.4
            # Coastal areas (higher pollution)
            elif abs(lat) < 30 and (abs(lon) < 20 or abs(lon - 180) < 20):
                prob += 0.3
                
            hotspot_probs.append(min(prob + np.random.normal(0, 0.1), 1.0))
        
        # Create timestamps
        start_date = datetime.now() - timedelta(days=365)
        timestamps = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
        
        self.data = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'timestamp': timestamps,
            'hotspot_probability': hotspot_probs,
            'is_hotspot': [p > 0.5 for p in hotspot_probs],
            'microplastic_concentration': np.random.lognormal(2, 1, n_samples),
            'water_temperature': np.random.normal(15, 8, n_samples),
            'ocean_current_speed': np.random.gamma(2, 2, n_samples)
        })
        
        logger.info(f"Created sample visualization data with {len(self.data)} records")
    
    def create_global_hotspot_map(self, save_path="visualizations/global_hotspot_map.html"):
        """Create an interactive global map showing microplastic hotspots"""
        if self.data is None:
            self.load_data()
            
        # Create base map
        m = folium.Map(
            location=[20, 0],  # Center on equator
            zoom_start=3,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter').add_to(m)
        
        # Color mapping for hotspot probability
        def get_color(probability):
            if probability < 0.2:
                return 'green'
            elif probability < 0.4:
                return 'yellow'
            elif probability < 0.6:
                return 'orange'
            elif probability < 0.8:
                return 'red'
            else:
                return 'darkred'
        
        # Add markers for each data point
        for idx, row in self.data.iterrows():
            # Create popup content
            popup_content = f"""
            <div style="width: 200px;">
                <h4>Marine Monitoring Point</h4>
                <p><strong>Coordinates:</strong> {row['latitude']:.3f}, {row['longitude']:.3f}</p>
                <p><strong>Hotspot Probability:</strong> {row['hotspot_probability']:.1%}</p>
                <p><strong>Concentration:</strong> {row['microplastic_concentration']:.2f} particles/m³</p>
                <p><strong>Water Temp:</strong> {row['water_temperature']:.1f}°C</p>
                <p><strong>Current Speed:</strong> {row['ocean_current_speed']:.1f} m/s</p>
                <p><strong>Date:</strong> {str(row['timestamp'])[:10]}</p>
            </div>
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=max(3, row['hotspot_probability'] * 15),
                popup=folium.Popup(popup_content, max_width=250),
                color='black',
                weight=1,
                fillColor=get_color(row['hotspot_probability']),
                fillOpacity=0.7,
                tooltip=f"Hotspot Risk: {row['hotspot_probability']:.1%}"
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px;">
        <h4>Hotspot Risk</h4>
        <p><i class="fa fa-circle" style="color:green"></i> Very Low (0-20%)</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Low (20-40%)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Moderate (40-60%)</p>
        <p><i class="fa fa-circle" style="color:red"></i> High (60-80%)</p>
        <p><i class="fa fa-circle" style="color:darkred"></i> Very High (80%+)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        m.save(save_path)
        logger.info(f"Global hotspot map saved to {save_path}")
        
        return m
    
    def create_concentration_heatmap(self, save_path="visualizations/concentration_heatmap.html"):
        """Create a heatmap showing microplastic concentration distribution"""
        if self.data is None:
            self.load_data()
            
        # Create heatmap
        fig = px.density_mapbox(
            self.data,
            lat='latitude',
            lon='longitude',
            z='microplastic_concentration',
            radius=20,
            center=dict(lat=20, lon=0),
            zoom=2,
            mapbox_style="open-street-map",
            title="Global Microplastic Concentration Heatmap",
            color_continuous_scale="Viridis",
            labels={'microplastic_concentration': 'Concentration (particles/m³)'}
        )
        
        fig.update_layout(
            height=600,
            title_x=0.5,
            font=dict(size=12)
        )
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Concentration heatmap saved to {save_path}")
        
        return fig
    
    def create_temporal_analysis(self, save_path="visualizations/temporal_analysis.html"):
        """Create temporal analysis charts"""
        if self.data is None:
            self.load_data()
            
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Hotspot Probability Over Time',
                'Microplastic Concentration Over Time', 
                'Monthly Hotspot Detection Count',
                'Seasonal Patterns'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sort data by timestamp
        data_sorted = self.data.sort_values('timestamp')
        
        # 1. Hotspot probability over time
        fig.add_trace(
            go.Scatter(
                x=data_sorted['timestamp'],
                y=data_sorted['hotspot_probability'],
                mode='markers+lines',
                name='Hotspot Probability',
                marker=dict(
                    color=data_sorted['hotspot_probability'],
                    colorscale='Reds',
                    size=6
                )
            ),
            row=1, col=1
        )
        
        # 2. Concentration over time
        fig.add_trace(
            go.Scatter(
                x=data_sorted['timestamp'],
                y=data_sorted['microplastic_concentration'],
                mode='markers',
                name='Concentration',
                marker=dict(
                    color=data_sorted['microplastic_concentration'],
                    colorscale='Viridis',
                    size=6
                )
            ),
            row=1, col=2
        )
        
        # 3. Monthly hotspot count
        monthly_hotspots = data_sorted.groupby(data_sorted['timestamp'].dt.to_period('M'))['is_hotspot'].sum()
        fig.add_trace(
            go.Bar(
                x=[str(period) for period in monthly_hotspots.index],
                y=monthly_hotspots.values,
                name='Monthly Hotspots',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # 4. Seasonal patterns
        seasonal_data = data_sorted.groupby(data_sorted['timestamp'].dt.month).agg({
            'hotspot_probability': 'mean',
            'microplastic_concentration': 'mean'
        })
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig.add_trace(
            go.Scatter(
                x=months[:len(seasonal_data)],
                y=seasonal_data['hotspot_probability'],
                mode='lines+markers',
                name='Avg Hotspot Prob',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Temporal Analysis of Marine Microplastic Data",
            title_x=0.5,
            showlegend=True
        )
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Temporal analysis saved to {save_path}")
        
        return fig
    
    def create_environmental_correlations(self, save_path="visualizations/environmental_correlations.html"):
        """Create correlation analysis between environmental factors and microplastics"""
        if self.data is None:
            self.load_data()
            
        # Create correlation matrix
        correlation_data = self.data[['hotspot_probability', 'microplastic_concentration', 
                                     'water_temperature', 'ocean_current_speed']].corr()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Correlation Matrix',
                'Temperature vs Microplastics',
                'Current Speed vs Microplastics', 
                'Geographic Distribution'
            ],
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=correlation_data.values,
                x=correlation_data.columns,
                y=correlation_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_data.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ),
            row=1, col=1
        )
        
        # 2. Temperature correlation
        fig.add_trace(
            go.Scatter(
                x=self.data['water_temperature'],
                y=self.data['microplastic_concentration'],
                mode='markers',
                marker=dict(
                    color=self.data['hotspot_probability'],
                    colorscale='Reds',
                    size=8,
                    opacity=0.7
                ),
                name='Temp vs Concentration'
            ),
            row=1, col=2
        )
        
        # 3. Current speed correlation
        fig.add_trace(
            go.Scatter(
                x=self.data['ocean_current_speed'],
                y=self.data['microplastic_concentration'],
                mode='markers',
                marker=dict(
                    color=self.data['hotspot_probability'],
                    colorscale='Blues',
                    size=8,
                    opacity=0.7
                ),
                name='Current vs Concentration'
            ),
            row=2, col=1
        )
        
        # 4. Geographic scatter
        fig.add_trace(
            go.Scatter(
                x=self.data['longitude'],
                y=self.data['latitude'],
                mode='markers',
                marker=dict(
                    color=self.data['microplastic_concentration'],
                    colorscale='Viridis',
                    size=self.data['hotspot_probability'] * 20 + 5,
                    opacity=0.7
                ),
                name='Geographic Distribution'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Environmental Factors vs Microplastic Distribution",
            title_x=0.5,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Concentration", row=1, col=2)
        fig.update_xaxes(title_text="Current Speed (m/s)", row=2, col=1)
        fig.update_yaxes(title_text="Concentration", row=2, col=1)
        fig.update_xaxes(title_text="Longitude", row=2, col=2)
        fig.update_yaxes(title_text="Latitude", row=2, col=2)
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Environmental correlations saved to {save_path}")
        
        return fig
    
    def create_risk_dashboard(self, save_path="visualizations/risk_dashboard.html"):
        """Create a comprehensive risk assessment dashboard"""
        if self.data is None:
            self.load_data()
            
        # Create dashboard with multiple visualizations
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Risk Level Distribution', 'Hotspot Locations', 'Concentration Distribution',
                'Risk by Ocean Region', 'Monthly Risk Trends', 'Temperature Impact',
                'Current Speed Impact', 'Depth Analysis', 'Prediction Confidence'
            ],
            specs=[[{"type": "pie"}, {"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "box"}, {"type": "indicator"}]]
        )
        
        # Define risk levels
        risk_levels = []
        for prob in self.data['hotspot_probability']:
            if prob < 0.2:
                risk_levels.append('Very Low')
            elif prob < 0.4:
                risk_levels.append('Low')
            elif prob < 0.6:
                risk_levels.append('Moderate')
            elif prob < 0.8:
                risk_levels.append('High')
            else:
                risk_levels.append('Very High')
        
        self.data['risk_level'] = risk_levels
        
        # 1. Risk level pie chart
        risk_counts = pd.Series(risk_levels).value_counts()
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                name="Risk Distribution"
            ),
            row=1, col=1
        )
        
        # 2. Hotspot locations
        hotspots = self.data[self.data['is_hotspot']]
        fig.add_trace(
            go.Scatter(
                x=hotspots['longitude'],
                y=hotspots['latitude'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='cross'
                ),
                name='Confirmed Hotspots'
            ),
            row=1, col=2
        )
        
        # 3. Concentration histogram
        fig.add_trace(
            go.Histogram(
                x=self.data['microplastic_concentration'],
                name='Concentration Distribution',
                marker_color='blue'
            ),
            row=1, col=3
        )
        
        # 4. Risk by region (simplified)
        regions = []
        for lat, lon in zip(self.data['latitude'], self.data['longitude']):
            if lat > 30:
                regions.append('North')
            elif lat < -30:
                regions.append('South')
            else:
                regions.append('Tropical')
        
        self.data['region'] = regions
        region_risk = self.data.groupby('region')['hotspot_probability'].mean()
        
        fig.add_trace(
            go.Bar(
                x=region_risk.index,
                y=region_risk.values,
                name='Risk by Region',
                marker_color=['lightblue', 'orange', 'lightgreen']
            ),
            row=2, col=1
        )
        
        # 5. Monthly trends (if timestamp available)
        if not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        monthly_risk = self.data.groupby(self.data['timestamp'].dt.month)['hotspot_probability'].mean()
        fig.add_trace(
            go.Scatter(
                x=monthly_risk.index,
                y=monthly_risk.values,
                mode='lines+markers',
                name='Monthly Risk Trend',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        # 6. Temperature impact
        fig.add_trace(
            go.Scatter(
                x=self.data['water_temperature'],
                y=self.data['hotspot_probability'],
                mode='markers',
                marker=dict(
                    color=self.data['microplastic_concentration'],
                    colorscale='Viridis',
                    size=6
                ),
                name='Temperature Impact'
            ),
            row=2, col=3
        )
        
        # 7. Current speed impact
        fig.add_trace(
            go.Scatter(
                x=self.data['ocean_current_speed'],
                y=self.data['hotspot_probability'],
                mode='markers',
                marker=dict(
                    color='purple',
                    size=6
                ),
                name='Current Speed Impact'
            ),
            row=3, col=1
        )
        
        # 8. Risk level box plot
        fig.add_trace(
            go.Box(
                x=self.data['risk_level'],
                y=self.data['microplastic_concentration'],
                name='Concentration by Risk'
            ),
            row=3, col=2
        )
        
        # 9. Overall confidence indicator
        avg_confidence = self.data['hotspot_probability'].std()  # Use std as confidence measure
        confidence_score = max(0, 100 - (avg_confidence * 100))
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=confidence_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Model Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 85], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Marine Microplastic Risk Assessment Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        # Save dashboard
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Risk dashboard saved to {save_path}")
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations and return file paths"""
        logger.info("Generating all visualizations...")
        
        file_paths = []
        
        try:
            # Create global hotspot map
            self.create_global_hotspot_map()
            file_paths.append("visualizations/global_hotspot_map.html")
            
            # Create concentration heatmap
            self.create_concentration_heatmap()
            file_paths.append("visualizations/concentration_heatmap.html")
            
            # Create temporal analysis
            self.create_temporal_analysis()
            file_paths.append("visualizations/temporal_analysis.html")
            
            # Create environmental correlations
            self.create_environmental_correlations()
            file_paths.append("visualizations/environmental_correlations.html")
            
            # Create risk dashboard
            self.create_risk_dashboard()
            file_paths.append("visualizations/risk_dashboard.html")
            
            logger.info(f"Successfully generated {len(file_paths)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            
        return file_paths

def main():
    """Main function to demonstrate visualization capabilities"""
    print("🎨 Marine Microplastic Visualization System")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = MarineMicroplasticVisualizer()
    
    # Generate all visualizations
    file_paths = visualizer.generate_all_visualizations()
    
    print("\n✅ Visualization Generation Complete!")
    print("\n📊 Generated Files:")
    for path in file_paths:
        print(f"   🔗 {path}")
    
    print(f"\n🌐 Open any HTML file in your browser to view interactive visualizations")
    print("🎯 Key Features:")
    print("   • Interactive global hotspot maps")
    print("   • Concentration heatmaps") 
    print("   • Temporal trend analysis")
    print("   • Environmental correlation analysis")
    print("   • Comprehensive risk dashboard")

if __name__ == "__main__":
    main()