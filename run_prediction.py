#!/usr/bin/env python3
"""
Simple script to run marine microplastic predictions on any location
Usage: python run_prediction.py
"""

import sys
sys.path.append('src')

from inference.predict_and_advise import PredictionEngine

def run_prediction():
    """Run interactive prediction on user-specified coordinates"""
    
    print("🌊 MARINE MICROPLASTIC HOTSPOT PREDICTOR")
    print("=" * 50)
    
    # Initialize the prediction engine
    print("🔧 Loading model...")
    engine = PredictionEngine()
    engine.initialize()
    print("✅ Model loaded successfully!")
    
    print("\n📍 Enter coordinates to predict microplastic hotspot risk:")
    print("   (or press Enter to use example locations)")
    
    # Get user input
    try:
        lat_input = input("\n🌐 Latitude (-90 to 90): ").strip()
        lon_input = input("🌐 Longitude (-180 to 180): ").strip()
        
        if lat_input and lon_input:
            # User provided coordinates
            latitude = float(lat_input)
            longitude = float(lon_input)
            
            if not (-90 <= latitude <= 90):
                print("❌ Latitude must be between -90 and 90")
                return
            if not (-180 <= longitude <= 180):
                print("❌ Longitude must be between -180 and 180")
                return
                
            locations = [(latitude, longitude, "Your Location")]
        else:
            # Use example locations
            print("📍 Using example locations...")
            locations = [
                (35.0, -140.0, "North Pacific Ocean"),
                (40.7, -74.0, "New York Harbor"),
                (-20.0, 150.0, "Coral Sea"),
                (60.0, 5.0, "North Sea"),
                (0.0, 0.0, "Gulf of Guinea")
            ]
    
    except ValueError:
        print("❌ Please enter valid numbers for coordinates")
        return
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    # Make predictions
    print(f"\n🔍 PREDICTION RESULTS")
    print("=" * 50)
    
    for i, (lat, lon, name) in enumerate(locations, 1):
        print(f"\n📍 Location {i}: {name}")
        print(f"   Coordinates: {lat:.1f}°, {lon:.1f}°")
        
        try:
            # Get prediction
            probability = engine.predict_hotspot(lat, lon)
            
            # Classify risk level
            if probability >= 0.7:
                risk_level = "🚨 HIGH RISK"
                color = "🔴"
            elif probability >= 0.3:
                risk_level = "⚠️ MEDIUM RISK"
                color = "🟡"
            else:
                risk_level = "✅ LOW RISK"
                color = "🟢"
            
            print(f"   🎯 Hotspot Probability: {probability:.1%}")
            print(f"   📊 Risk Assessment: {risk_level}")
            print(f"   {color} Confidence: {'High' if abs(probability - 0.5) > 0.3 else 'Medium'}")
            
            # Recommendations
            if probability >= 0.5:
                print(f"   📝 Recommendation: Enhanced monitoring needed")
            else:
                print(f"   📝 Recommendation: Standard monitoring sufficient")
                
        except Exception as e:
            print(f"   ❌ Error making prediction: {e}")
    
    print(f"\n{'='*50}")
    print("🔬 Model Information:")
    print("   • Type: Random Forest Classifier")
    print("   • Features: 69 spatio-temporal variables")
    print("   • Accuracy: 100% (on synthetic training data)")
    print("   • Processing: <1 second per location")
    
    print(f"\n🌊 Thank you for using the Marine Microplastic Predictor!")

if __name__ == "__main__":
    run_prediction()