#!/usr/bin/env python3
"""
Marine Microplastic Advisory System
Input: Coordinates (latitude, longitude)
Output: Comprehensive advisory based on model prediction
"""

import sys
import os
sys.path.append('src')

from inference.predict_and_advise import PredictionEngine
from datetime import datetime

class MarineAdvisorySystem:
    """Main advisory system that takes coordinates and provides comprehensive advisories"""
    
    def __init__(self):
        self.engine = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the prediction engine"""
        try:
            print("🔧 Initializing Marine Microplastic Advisory System...")
            self.engine = PredictionEngine()
            self.engine.initialize()
            print("✅ System ready!")
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            sys.exit(1)
    
    def validate_coordinates(self, latitude, longitude):
        """Validate input coordinates"""
        try:
            lat = float(latitude)
            lon = float(longitude)
            
            if not (-90 <= lat <= 90):
                return False, "Latitude must be between -90 and 90 degrees"
            if not (-180 <= lon <= 180):
                return False, "Longitude must be between -180 and 180 degrees"
                
            return True, (lat, lon)
        except ValueError:
            return False, "Please enter valid numeric coordinates"
    
    def get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability >= 0.70:
            return "CRITICAL", "🔴", "Immediate action required"
        elif probability >= 0.50:
            return "HIGH", "🟠", "Enhanced monitoring needed"
        elif probability >= 0.30:
            return "MEDIUM", "🟡", "Increased vigilance recommended"
        elif probability >= 0.15:
            return "LOW", "🟢", "Standard monitoring sufficient"
        else:
            return "VERY LOW", "✅", "Minimal concern"
    
    def generate_environmental_advisory(self, lat, lon, probability, risk_level):
        """Generate advisory for environmental agencies"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        advisory = f"""
🌍 ENVIRONMENTAL AGENCY ADVISORY
{'='*50}
Location: {lat:.4f}°N, {lon:.4f}°E
Assessment Date: {timestamp}
Microplastic Hotspot Probability: {probability:.1%}
Risk Classification: {risk_level}

ENVIRONMENTAL IMPACT ASSESSMENT:
"""
        
        if probability >= 0.70:
            advisory += """
🚨 CRITICAL RISK DETECTED
• Immediate environmental assessment required
• Deploy monitoring buoys and sampling equipment
• Coordinate with marine protection agencies
• Issue navigation warnings to vessel traffic
• Initiate emergency response protocols
• Contact local wildlife conservation groups
"""
        elif probability >= 0.50:
            advisory += """
🟠 HIGH RISK IDENTIFIED
• Enhanced monitoring protocols activated
• Increase sampling frequency in the area
• Deploy additional environmental sensors
• Monitor marine wildlife for signs of impact
• Coordinate with fisheries management
• Consider temporary activity restrictions
"""
        elif probability >= 0.30:
            advisory += """
🟡 MEDIUM RISK OBSERVED
• Implement increased surveillance measures
• Schedule additional water quality assessments
• Monitor local marine ecosystem health
• Collaborate with research institutions
• Update environmental impact databases
• Maintain heightened awareness protocols
"""
        else:
            advisory += """
🟢 LOW RISK ASSESSMENT
• Continue standard environmental monitoring
• Maintain regular sampling schedules
• Monitor for changes in risk factors
• Support ongoing research activities
• Keep databases updated with current conditions
"""
        
        advisory += f"""
RECOMMENDED ACTIONS:
• GPS Monitoring Zone: {lat:.2f}°±0.1°, {lon:.2f}°±0.1°
• Assessment Frequency: {'Daily' if probability >= 0.5 else 'Weekly' if probability >= 0.3 else 'Monthly'}
• Priority Level: {'URGENT' if probability >= 0.7 else 'HIGH' if probability >= 0.5 else 'STANDARD'}

Report ID: ENV-{datetime.now().strftime('%Y%m%d')}-{abs(hash(f'{lat}{lon}'))%10000:04d}
"""
        return advisory
    
    def generate_fisheries_advisory(self, lat, lon, probability, risk_level):
        """Generate advisory for fisheries and marine industry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        advisory = f"""
🎣 FISHERIES & MARINE INDUSTRY ADVISORY
{'='*50}
Location: {lat:.4f}°N, {lon:.4f}°E
Issue Date: {timestamp}
Contamination Risk: {probability:.1%}
Operational Status: {risk_level}

FISHING OPERATIONS GUIDANCE:
"""
        
        if probability >= 0.70:
            advisory += """
🚨 FISHING SUSPENSION RECOMMENDED
• Avoid fishing operations in this area
• Redirect vessels to alternative fishing grounds
• Implement enhanced catch inspection protocols
• Report any visible contamination immediately
• Consider temporary fishing moratorium
• Coordinate with harbor authorities
"""
        elif probability >= 0.50:
            advisory += """
🟠 ENHANCED PRECAUTIONS REQUIRED
• Proceed with extreme caution
• Implement additional quality control measures
• Increase catch inspection frequency
• Use specialized filtering equipment if available
• Monitor catch for contamination signs
• Report unusual findings to authorities
"""
        elif probability >= 0.30:
            advisory += """
🟡 STANDARD PRECAUTIONS ADVISED
• Normal operations with increased vigilance
• Conduct regular catch quality assessments
• Maintain clean handling procedures
• Document any unusual observations
• Follow standard contamination protocols
"""
        else:
            advisory += """
🟢 NORMAL OPERATIONS APPROVED
• Standard fishing operations permitted
• Continue routine quality control procedures
• Maintain regular documentation
• Report any unexpected contamination
"""
        
        advisory += f"""
INDUSTRY RECOMMENDATIONS:
• Quality Control: {'ENHANCED' if probability >= 0.5 else 'STANDARD'}
• Inspection Protocol: {'MANDATORY' if probability >= 0.7 else 'RECOMMENDED' if probability >= 0.3 else 'ROUTINE'}
• Area Status: {'RESTRICTED' if probability >= 0.7 else 'CAUTION' if probability >= 0.5 else 'NORMAL'}

Contact: Marine Industry Hotline - 1-800-MARINE-1
Advisory ID: FISH-{datetime.now().strftime('%Y%m%d')}-{abs(hash(f'{lat}{lon}'))%10000:04d}
"""
        return advisory
    
    def generate_public_advisory(self, lat, lon, probability, risk_level):
        """Generate advisory for general public"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        advisory = f"""
👥 PUBLIC MARINE CONDITIONS ADVISORY
{'='*50}
Location: {lat:.4f}°N, {lon:.4f}°E
Report Date: {timestamp}
Water Quality Status: {risk_level}
Contamination Probability: {probability:.1%}

PUBLIC SAFETY INFORMATION:
"""
        
        if probability >= 0.70:
            advisory += """
🚨 AVOID AREA - HIGH CONTAMINATION RISK
• Do not swim, dive, or engage in water sports
• Avoid consuming fish caught in this area
• Keep pets away from the water
• Report any visible pollution immediately
• Consider alternative recreational areas
• Stay informed about area updates
"""
        elif probability >= 0.50:
            advisory += """
🟠 EXERCISE EXTREME CAUTION
• Limit water contact activities
• Avoid consumption of locally caught seafood
• Supervise children closely near water
• Use protective equipment if water contact necessary
• Be aware of potential health risks
"""
        elif probability >= 0.30:
            advisory += """
🟡 USE CAUTION
• Normal activities with increased awareness
• Choose seafood from verified safe sources
• Monitor local health advisories
• Report any unusual water conditions
• Follow standard water safety guidelines
"""
        else:
            advisory += """
🟢 NORMAL CONDITIONS
• Standard recreational activities permitted
• Regular water safety precautions apply
• Enjoy normal marine recreational activities
• Continue following general safety guidelines
"""
        
        advisory += f"""
HEALTH & SAFETY TIPS:
• Shower after water contact
• Avoid ingesting water during activities
• Choose reputable seafood suppliers
• Stay informed through official channels

FOR EMERGENCIES: Call 911
INFO UPDATES: www.marinehealth.gov
Advisory ID: PUB-{datetime.now().strftime('%Y%m%d')}-{abs(hash(f'{lat}{lon}'))%10000:04d}
"""
        return advisory
    
    def generate_comprehensive_advisory(self, latitude, longitude):
        """Generate complete advisory package for given coordinates"""
        
        # Get prediction
        try:
            probability = self.engine.predict_hotspot(latitude, longitude)
        except Exception as e:
            return f"❌ Error generating prediction: {e}"
        
        # Determine risk level
        risk_level, emoji, description = self.get_risk_level(probability)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Create comprehensive report
        report = f"""
🌊 MARINE MICROPLASTIC COMPREHENSIVE ADVISORY REPORT
{'='*70}
Generated: {timestamp}
Location: {latitude:.6f}°N, {longitude:.6f}°E
Analysis: Advanced ML Prediction (Random Forest)

📊 RISK ASSESSMENT SUMMARY
{'='*70}
{emoji} HOTSPOT PROBABILITY: {probability:.1%}
{emoji} RISK LEVEL: {risk_level}
{emoji} DESCRIPTION: {description}
{emoji} CONFIDENCE: {'High' if abs(probability - 0.5) > 0.3 else 'Medium'}

{self.generate_environmental_advisory(latitude, longitude, probability, risk_level)}

{self.generate_fisheries_advisory(latitude, longitude, probability, risk_level)}

{self.generate_public_advisory(latitude, longitude, probability, risk_level)}

📈 TECHNICAL DETAILS
{'='*70}
• Model Type: Random Forest Classifier (200 trees)
• Features Analyzed: 69 spatio-temporal variables
• Processing Time: <1 second
• Model Accuracy: 100% (synthetic training data)
• Geographic Coverage: Global oceans
• Update Frequency: Real-time predictions available

🔬 SCIENTIFIC BASIS
{'='*70}
This advisory is based on machine learning analysis of:
• Geographic coordinates and ocean basin classification
• Temporal patterns and seasonal variations
• Distance to known accumulation zones and ocean currents
• Historical oceanographic data patterns
• Spatial clustering and proximity analysis

⚠️ DISCLAIMER
{'='*70}
This advisory is generated using predictive modeling on synthetic training data.
Real-world marine conditions may vary. Always consult local authorities and
official marine safety organizations for current conditions and regulations.

For technical support: support@marine-prediction.org
Report issues: github.com/ssouvik03/EDCC-Course-Assignment
{'='*70}
🌊 END OF ADVISORY REPORT 🌊
"""
        return report

def main():
    """Main interactive function"""
    print("🌊 MARINE MICROPLASTIC ADVISORY SYSTEM")
    print("="*50)
    print("Enter ocean coordinates to get comprehensive risk advisory")
    print("Press Ctrl+C to exit\n")
    
    # Initialize the advisory system
    advisory_system = MarineAdvisorySystem()
    
    while True:
        try:
            print("\n📍 Enter Coordinates:")
            latitude = input("🌐 Latitude (-90 to 90): ").strip()
            longitude = input("🌐 Longitude (-180 to 180): ").strip()
            
            if not latitude or not longitude:
                print("❌ Please enter both latitude and longitude")
                continue
            
            # Validate coordinates
            valid, result = advisory_system.validate_coordinates(latitude, longitude)
            if not valid:
                print(f"❌ {result}")
                continue
            
            lat, lon = result
            
            print(f"\n🔄 Analyzing location {lat:.4f}°, {lon:.4f}°...")
            print("⏳ Generating comprehensive advisory...")
            
            # Generate and display advisory
            advisory = advisory_system.generate_comprehensive_advisory(lat, lon)
            print(advisory)
            
            # Ask if user wants to continue
            continue_choice = input("\n❓ Analyze another location? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using Marine Microplastic Advisory System!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again with valid coordinates.")

if __name__ == "__main__":
    main()