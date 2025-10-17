"""
Simple Visualization Demo for Marine Microplastic Framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_simple_demo():
    """Create a simple but comprehensive visualization demo"""
    print("🎨 Creating Marine Microplastic Visualizations...")
    
    # Load the processed data
    data_path = "data/processed_marine_microplastics.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Standardize column names
        df.columns = df.columns.str.lower()
        print(f"✅ Loaded {len(df)} records")
        print(f"📊 Columns: {list(df.columns)}")
    else:
        print("❌ Data file not found, creating sample data...")
        return
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. BASIC STATISTICS PLOT
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🌊 Marine Microplastic Data Overview', fontsize=16)
    
    # Concentration distribution
    axes[0, 0].hist(df['concentration'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Microplastic Concentration Distribution')
    axes[0, 0].set_xlabel('Concentration (particles/m³)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Hotspot distribution
    hotspot_counts = df['is_hotspot'].value_counts()
    axes[0, 1].pie(hotspot_counts.values, labels=['Non-Hotspot', 'Hotspot'], 
                   autopct='%1.1f%%', colors=['lightblue', 'red'])
    axes[0, 1].set_title('Hotspot vs Non-Hotspot Distribution')
    
    # Geographic scatter
    scatter = axes[1, 0].scatter(df['longitude'], df['latitude'], 
                                c=df['concentration'], cmap='Reds', alpha=0.6)
    axes[1, 0].set_title('Geographic Distribution of Microplastics')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(scatter, ax=axes[1, 0], label='Concentration')
    
    # Ocean distribution
    if 'oceans' in df.columns:
        ocean_counts = df['oceans'].value_counts()
        axes[1, 1].bar(ocean_counts.index, ocean_counts.values, color='lightgreen')
        axes[1, 1].set_title('Data Points by Ocean')
        axes[1, 1].set_xlabel('Ocean')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/basic_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. INTERACTIVE FOLIUM MAP
    print("🗺️ Creating interactive map...")
    
    # Create base map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    # Add points to map
    for idx, row in df.sample(min(500, len(df))).iterrows():  # Sample for performance
        color = 'red' if row['is_hotspot'] else 'blue'
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=max(3, row['concentration']/5),
            popup=f"""
            <b>Marine Data Point</b><br>
            Coordinates: {row['latitude']:.2f}, {row['longitude']:.2f}<br>
            Concentration: {row['concentration']:.2f}<br>
            Ocean: {row.get('oceans', 'Unknown')}<br>
            Hotspot: {'Yes' if row['is_hotspot'] else 'No'}
            """,
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 80px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px;">
    <h4>Legend</h4>
    <p><i class="fa fa-circle" style="color:red"></i> Hotspot</p>
    <p><i class="fa fa-circle" style="color:blue"></i> Normal</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save('visualizations/interactive_map.html')
    print("✅ Interactive map saved: visualizations/interactive_map.html")
    
    # 3. PLOTLY INTERACTIVE CHARTS
    print("📈 Creating interactive charts...")
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Concentration by Ocean', 'Hotspot Distribution', 
                       'Geographic Heatmap', 'Temporal Analysis'],
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Concentration by ocean
    if 'oceans' in df.columns:
        ocean_conc = df.groupby('oceans')['concentration'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=ocean_conc['oceans'], y=ocean_conc['concentration'], 
                   name='Avg Concentration', marker_color='lightblue'),
            row=1, col=1
        )
    
    # Hotspot pie chart
    hotspot_counts = df['is_hotspot'].value_counts()
    fig.add_trace(
        go.Pie(labels=['Non-Hotspot', 'Hotspot'], values=hotspot_counts.values,
               name="Hotspot Distribution", marker_colors=['lightblue', 'red']),
        row=1, col=2
    )
    
    # Geographic scatter
    fig.add_trace(
        go.Scatter(x=df['longitude'], y=df['latitude'],
                   mode='markers',
                   marker=dict(color=df['concentration'], colorscale='Reds',
                              size=6, opacity=0.7,
                              colorbar=dict(title="Concentration")),
                   name='Geographic Distribution'),
        row=2, col=1
    )
    
    # Temporal analysis (if date column exists)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        daily_conc = df.groupby(df['date'].dt.date)['concentration'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=daily_conc['date'], y=daily_conc['concentration'],
                       mode='lines', name='Daily Avg Concentration',
                       line=dict(color='green')),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="🌊 Marine Microplastic Interactive Dashboard")
    fig.write_html('visualizations/interactive_dashboard.html')
    print("✅ Interactive dashboard saved: visualizations/interactive_dashboard.html")
    
    # 4. SUMMARY STATISTICS
    print("\n📊 DATA SUMMARY")
    print("=" * 50)
    print(f"Total data points: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Coordinate range:")
    print(f"  Latitude: {df['latitude'].min():.1f}° to {df['latitude'].max():.1f}°")
    print(f"  Longitude: {df['longitude'].min():.1f}° to {df['longitude'].max():.1f}°")
    print(f"Concentration range: {df['concentration'].min():.2f} to {df['concentration'].max():.2f}")
    print(f"Hotspot percentage: {df['is_hotspot'].mean():.1%}")
    
    if 'oceans' in df.columns:
        print(f"\nOcean distribution:")
        for ocean, count in df['oceans'].value_counts().items():
            print(f"  {ocean}: {count:,} points ({count/len(df):.1%})")
    
    print(f"\n🎉 All visualizations created successfully!")
    print(f"📁 Files saved in 'visualizations/' directory:")
    print(f"   📊 basic_statistics.png")
    print(f"   🗺️ interactive_map.html")
    print(f"   📈 interactive_dashboard.html")
    print(f"\n🌐 Open HTML files in your browser for interactive exploration!")

if __name__ == "__main__":
    create_simple_demo()